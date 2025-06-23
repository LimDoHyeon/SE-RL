#!/usr/bin/env python
"""Trainer rewritten for DDP-friendly training.
Removes hard-coded .cuda() / device selection and relies on the caller to
wrap the model with DistributedDataParallel (DDP) and to place it on the
correct device.  All tensors are moved to the same device as the model at
runtime so the code now works on CPU, single-GPU, or multi-GPU DDP
without further changes.
"""
from __future__ import annotations

import logging, time, sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_  # type: ignore
import torch.distributed as dist

from utils.util import check_parameters
from model.loss import Loss, mlloss
try:                               # 터미널 / 노트북 어디서든 동작
    from tqdm.auto import tqdm
except ImportError:                # tqdm < 4.65 호환
    from tqdm import tqdm
from utils.metrics import pesq_wrapper as pesq


class Trainer:
    """Generic training/validation loop wrapper, DDP‑safe."""

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
        opt,
        *,
        wandb_run=None,
    ) -> None:
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Core objects
        self.model = model  # may already be a DDP wrapper
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_run = wandb_run

        # Hyper‑parameters / misc flags
        self.total_epoch: int = opt.epochs
        self.early_stop: int = opt.early_stop
        self.print_freq: int = opt.print_freq
        self.clip_norm: float = opt.max_norm if opt.max_norm else 0.0
        self.save_path: Path = Path(opt.save_folder)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.resume_state: int = opt.resume_state
        self.weights: float = opt.weights
        self.episod: int = opt.episod
        self.is_main = (not dist.is_initialized()) or dist.get_rank() == 0

        # Device inferred from the model
        self.device: torch.device = next(model.parameters()).device

        # Logging
        self.logger = logging.getLogger(opt.logger_name)
        self.logger.info(
            "Model parameters: %.3f Mb", check_parameters(self.model)
        )

        # Resume checkpoint if requested
        if self.resume_state and getattr(opt, "resume_path", ""):
            self._load_checkpoint(opt.resume_path, partial=(self.resume_state == 2))
        elif self.resume_state:
            self.logger.warning("resume_state set but resume_path is empty—skipping checkpoint load.")

        # Internal trackers
        self.cur_epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        best_val_loss = float('inf')
        no_improve = 0
        while self.cur_epoch < self.total_epoch:
            self.cur_epoch += 1
            if hasattr(self.train_dataloader.dataset, "reseed_subset"):
                self.train_dataloader.dataset.reseed_subset(self.cur_epoch - 1)

            # 1) 학습
            tr_loss, tr_snr = self._train_one_epoch(self.cur_epoch)

            # 2) 검증
            val_loss, val_snr, val_pesq = self._validate(self.cur_epoch)

            # 3) WandB 로깅
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": self.cur_epoch,
                    "train_loss": tr_loss,
                    "train_snr": tr_snr,
                    "val_loss": val_loss,
                    "val_snr": val_snr,
                    "val_pesq": val_pesq,
                })

            # 4) Early stopping: 10 epoch 이하에서는 카운트하지 않음
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                self._save_checkpoint(self.cur_epoch, tr_loss, val_loss)
            else:
                # 에포크 10 이후부터만 no_improve 증가
                if self.cur_epoch > 10:
                    no_improve += 1
                if no_improve >= self.early_stop:
                    self.logger.info("Early stopping at epoch %d", self.cur_epoch)
                    break

    # Compatibility wrappers (for older scripts)
    # --------------------------------------------------------------

    def train(self, epoch: int) -> tuple[float, float]:
        """한 epoch 학습 수행 (tqdm으로 콘솔에 바로 진행률 표시)"""
        self.logger.info("Start training epoch %d", epoch)
        self.model.train()

        epoch_loss, epoch_snr = 0.0, 0.0
        start_time = time.time()

        # tqdm으로 데이터로더를 래핑하면 루프 돌면서 콘솔에 자동으로 프로그레스 바가 찍힙니다.
        loop = tqdm(
            self.train_dataloader,
            desc=f"[Train][Epoch {epoch}]",
            unit="batch",
            position=0,  # 여러 bar 겹칠 때 첫 줄 고정
            dynamic_ncols=True,
            disable=not self.is_main,  # rank0 만 표시
            file=sys.stdout,
        )
        max_steps = len(self.train_dataloader)
        for step, (data_noisy, data_clean) in enumerate(loop, 1):
            if step > max_steps:
                break
            else:
                data_noisy = data_noisy.to(self.device, non_blocking=True)
                data_clean = data_clean.to(self.device, non_blocking=True)

            if self.episod > 0:
                # RL 모드
                inputs = data_noisy
                total_rewards, total_action_prob = [], []
                total_outs, total_outputs = [], []

                for _ in range(self.episod):
                    outs, outputs, act_prob = self.model(inputs)
                    total_outs.append(outs)
                    total_outputs.append(outputs)
                    reward = mlloss(inputs, outputs.detach(), data_clean)
                    total_rewards.append(reward)
                    total_action_prob.append(act_prob)
                    inputs = outputs.detach()

                for i in range(self.episod):
                    self.optimizer.zero_grad(set_to_none=True)
                    G = total_rewards[i]
                    base_gamma = 0.1
                    for j in range(i + 1, self.episod):
                        gamma = base_gamma
                        G = total_rewards[i]
                        for j in range(i+1, self.episod):
                            G = G + gamma * total_rewards[j]
                            gamma = gamma * base_gamma

                    action_prob = torch.sum(total_action_prob[i], dim=1)
                    state_value = -1.0 * torch.mean(G * action_prob)

                    # debugging
                    outs, outputs, act_prob = self.model(data_noisy)
                    print("outs min/max/any NaN:", outs.min().item(), outs.max().item(), torch.isnan(outs).any())
                    print("outputs min/max/any NaN:", outputs.min().item(), outputs.max().item(),
                          torch.isnan(outputs).any())
                    print("act_prob min/max/any NaN:", act_prob.min().item(), act_prob.max().item(),
                          torch.isnan(act_prob).any())

                    loss_dist_rl, _ = Loss(total_outs[i], total_outputs[i], data_clean)
                    loss = state_value + self.weights * loss_dist_rl
                    loss.backward()

                    if self.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()

                loss_dist, loss_snr_rl = Loss(total_outs[-1], total_outputs[-1], data_clean)
                epoch_loss += loss_dist.item()
                epoch_snr += loss_snr_rl.item()

            else:
                # Supervised-only 모드
                outs, outputs, _ = self.model(data_noisy)
                loss_dist, loss_snr_sl = Loss(outs, outputs, data_clean)

                self.optimizer.zero_grad(set_to_none=True)
                loss_dist.backward()
                if self.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._unwrap(self.model).parameters(), self.clip_norm)
                self.optimizer.step()

                epoch_loss += loss_dist.item()
                epoch_snr += loss_snr_sl.item()

            # tqdm 프로그레스 바에 현재 평균 loss, snr을 함께 보여주도록 설정
            avg_loss = epoch_loss / step
            avg_snr = epoch_snr / step
            loop.set_postfix({"loss": f"{avg_loss:.4f}", "snr": f"{avg_snr:.2f}"})

        epoch_loss /= len(self.train_dataloader)
        epoch_snr /= len(self.train_dataloader)
        self.logger.info(
            f"Finished Epoch {epoch:03d}  ─ loss: {epoch_loss:.4f},  snr: {epoch_snr:.4f}dB"
        )
        return epoch_loss, epoch_snr

    def validation(self, epoch: int) -> tuple[float, float]:
        """한 epoch 검증 수행 (tqdm 적용 버전)"""
        self.logger.info("Start Validation epoch %d", epoch)
        self.model.eval()

        val_loss, val_snr = 0.0, 0.0
        start_time = time.time()

        # ── `tqdm`으로 데이터로더를 래핑 ──
        loop = tqdm(
            self.val_dataloader,
            desc=f"[Valid][Epoch {epoch}]",
            unit="batch",
            position=0,  # 여러 bar 겹칠 때 첫 줄 고정
            dynamic_ncols=True,
            disable=not self.is_main,  # rank0 만 표시
            file=sys.stdout,
        )
        with torch.no_grad():
            # for step, (data_noisy, data_clean) in enumerate(loop, 1):
            for step, (data_noisy, data_clean, *__) in enumerate(loop, 1):
                data_noisy = data_noisy.to(self.device, non_blocking=True)
                data_clean = data_clean.to(self.device, non_blocking=True)

                if self.episod > 0:
                    inputs = data_noisy
                    for _ in range(self.episod):
                        outs, outputs, _ = self.model(inputs)
                        inputs = outputs
                    loss_dist, loss_snr = Loss(outs, inputs, data_clean)
                else:
                    outs, outputs, _ = self.model(data_noisy)
                    loss_dist, loss_snr = Loss(outs, outputs, data_clean)

                val_loss += loss_dist.item()
                val_snr += loss_snr.item()

                # ── tqdm 막대 갱신 ──
                loop.set_postfix({"val_loss": val_loss / step, "val_snr": val_snr / step})

        val_loss /= len(self.val_dataloader)
        val_snr /= len(self.val_dataloader)
        self.logger.info(f"[E{epoch:03d}] VAL loss={val_loss:.4f}  snr={val_snr:.2f} dB")
        return val_loss, val_snr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        epoch_loss, epoch_snr = 0.0, 0.0
        start_time = time.time()

        loop = tqdm(
            self.train_dataloader,
            desc=f"[Train][E{epoch:03d}]",
            unit="batch",
            ascii=True,
            dynamic_ncols=True,
            disable=False,  # DDP면 rank0만 True로 두세요
            file=sys.stdout,
        )

        # for step, (data_noisy, data_clean) in enumerate(self.train_dataloader, 1):
        for step, (data_noisy, data_clean, *__) in enumerate(loop, 1):
            data_noisy = data_noisy.to(self.device, non_blocking=True)
            data_clean = data_clean.to(self.device, non_blocking=True)

            # RL 혹은 Supervised 모드 분기
            if self.episod > 0:
                # ── Reinforcement-style loop (episod > 0) ──
                inputs = data_noisy
                total_rewards, total_action_prob = [], []
                total_outs, total_outputs = [], []

                for _ in range(self.episod):
                    outs, outputs, act_prob = self.model(inputs)
                    total_outs.append(outs)
                    total_outputs.append(outputs)
                    # 보상 계산
                    reward = mlloss(inputs, outputs.detach(), data_clean)
                    total_rewards.append(reward)
                    total_action_prob.append(act_prob)
                    inputs = outputs.detach()

                # 에피소드별로 backward & optimization
                for i in range(self.episod):
                    self.optimizer.zero_grad(set_to_none=True)

                    # 누적 보상 G 계산 (할인율 gamma = 0.1)
                    G = total_rewards[i]
                    gamma = 0.1
                    for j in range(i + 1, self.episod):
                        G = G + gamma * total_rewards[j]
                        gamma *= gamma

                    # 정책 손실(state value)
                    # action_prob = 0.0001 * torch.sum(total_action_prob[i], dim=1)
                    action_prob = total_action_prob[i]  # [B]
                    state_value = -1.0 * torch.mean(G * action_prob)

                    # 왜곡 손실(signal distortion loss)
                    loss_dist, loss_snr_rl = Loss(total_outs[i], total_outputs[i], data_clean)

                    # 최종 손실: 정책 손실 + 가중치 * 왜곡 손실
                    loss = state_value + self.weights * loss_dist
                    loss.backward()

                    if self.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._unwrap(self.model).parameters(), self.clip_norm)
                    self.optimizer.step()

                # 마지막 에피소드의 왜곡 손실을 로그에 더함
                loss_dist, loss_snr_rl = Loss(total_outs[-1], total_outputs[-1], data_clean)
                epoch_loss += loss_dist.item()
                epoch_snr += loss_snr_rl.item()

                # ── CHG: 현재 평균을 bar에 반영 ──
                loop.set_postfix(
                    loss=f"{epoch_loss / step:.4f}",
                    snr=f"{epoch_snr / step:.2f}"
                )
            else:
                # ── Supervised-only 모드 (episod == 0) ──
                # 한 번만 forward → loss 계산 → backward → step
                outs, outputs, _ = self.model(data_noisy)
                loss_dist, loss_snr_sl = Loss(outs, outputs, data_clean)

                self.optimizer.zero_grad(set_to_none=True)
                loss_dist.backward()
                if self.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._unwrap(self.model).parameters(), self.clip_norm)
                self.optimizer.step()

                # 로깅용으로 누적
                epoch_loss += loss_dist.item()
                epoch_snr += loss_snr_sl.item()

            # 일정 주기마다 로그 출력
            if step % self.print_freq == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / step
                avg_snr = epoch_snr / step
                self.logger.info(
                    f"[Epoch {epoch:03d} Step {step:04d}] loss={avg_loss:.4f} snr={avg_snr:.2f}  time={elapsed / 60:.1f}min"
                )

        # epoch 당 평균 loss, snr 계산 후 반환
        epoch_loss /= len(self.train_dataloader)
        epoch_snr /= len(self.train_dataloader)
        self.logger.info(
            f"Finished Epoch {epoch:03d}  ─ loss: {epoch_loss:.4f},  snr: {epoch_snr:.4f}dB"
        )
        return epoch_loss, epoch_snr

    def _validate(self, epoch: int):
        self.model.eval()
        total_loss, total_snr, total_pesq = 0.0, 0.0, 0.0

        loop = tqdm(
            self.val_dataloader,
            desc=f"[Valid][E{epoch:03d}]",
            unit="batch",
            ascii=True,
            dynamic_ncols=True,
            disable=False,
            file=sys.stdout,
        )

        with torch.inference_mode():
            # for noisy, clean in self.val_dataloader:
            for noisy, clean, *__ in self.val_dataloader:
                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)

                outs, enhanced, _ = self.model(noisy)
                # RMSE = Loss 측정
                loss_dist, loss_snr_batch = Loss(outs, enhanced, clean)
                total_loss += loss_dist.item()
                total_snr += loss_snr_batch.item()

                # PESQ 계산
                enh_1d      = enhanced.detach().cpu().squeeze(1)
                clean_1d    = clean.detach().cpu().squeeze(1)
                batch_pesq  = pesq(enh_1d, clean_1d).mean().item()
                total_pesq += batch_pesq

        count = max(1, len(self.val_dataloader))
        avg_loss = total_loss / count
        avg_snr  = total_snr  / count
        avg_pesq = total_pesq / count

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.logger.info(
                "[Epoch %03d] VAL LOSS(RMSE)=%.4f  PESQ=%.2f  SNR=%.2f dB",
                epoch, avg_loss, avg_pesq, avg_snr
            )
        loop.set_postfix(
            rmse=f"{avg_loss:.4f}",
            snr=f"{avg_snr:.2f}"
        )
        return avg_loss, avg_snr, avg_pesq

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str | Path, *, partial: bool = False):
        ckpt = torch.load(path, map_location="cpu")
        self.cur_epoch = ckpt.get("epoch", 0)
        model_state = ckpt["model_state_dict"]
        if partial:
            curr_state = self._unwrap(self.model).state_dict()
            model_state = {k: v for k, v in model_state.items() if k in curr_state}
            curr_state.update(model_state)
            self._unwrap(self.model).load_state_dict(curr_state)
        else:
            self._unwrap(self.model).load_state_dict(model_state)
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.logger.info("Resumed from %s (epoch %d)", path, self.cur_epoch)

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        state = {
            "epoch": epoch,
            "model_state_dict": self._unwrap(self.model).state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        ckpt_name = f"nnet_iter{epoch}_trloss{train_loss:.4f}_valoss{val_loss:.4f}.pt"
        torch.save(state, str(self.save_path / ckpt_name))
        self.logger.info("Checkpoint saved: %s", ckpt_name)

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """Return underlying model if wrapped by DDP/DataParallel."""
        return model.module if hasattr(model, "module") else model