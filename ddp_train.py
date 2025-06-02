#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed-Data-Parallel entry script for SE-RL project
2025/06 – Do-Hyeon SERL
"""
import os, argparse, logging, time, math
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader.dataloader import AudioDataset
from model.modules import Base_model
import trainingmodule          # 그대로 재활용
import wandb

# ---------- Argument ---------- #
parser = argparse.ArgumentParser()
# (원래 인자 + DDP 전용)
parser.add_argument("--sample_rate", type=int, default=16_000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs",  type=int, default=300)
parser.add_argument("--lr",      type=float, default=1e-3)
parser.add_argument("--save_folder", default="exp/ddp_ckpt")
parser.add_argument("--experiment_name", default="SERL-DDP")
# 기존 train.py 의 나머지 인자들도 그대로 추가해 두면 됩니다 …
# ------------------------------------------------------------- #

def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format=f"[%(asctime)s][RANK{rank}] %(levelname)s ▸ %(message)s",
        datefmt="%H:%M:%S")
    return logging.getLogger(__name__)

def init_distributed():
    """Torch 2.0 의 torchrun 환경 변수를 읽어서 초기화"""
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = dist.get_world_size()
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

def average_all(tensor, world_size):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor

def main(args):
    local_rank, rank, world = init_distributed()
    logger = setup_logging(rank)
    device = torch.device(f"cuda:{local_rank}")

    # -------------- WandB (rank 0 만) -------------- #
    if rank == 0:
        wandb_run = wandb.init(
            project="SE-RL",
            name=args.experiment_name,
            config=vars(args))
    else:
        wandb_run = None

    # -------------- Datasets & Sampler -------------- #
    train_ds = AudioDataset(
        noisy_root=args.train_noisy_data_path,
        clean_root=args.train_clean_data_path,
        list_file=args.train_file,
        sample_rate=args.sample_rate)
    val_ds   = AudioDataset(
        noisy_root=args.valid_noisy_data_path,
        clean_root=args.valid_clean_data_path,
        list_file=args.valid_file,
        sample_rate=args.sample_rate)

    train_sampler = distributed.DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_sampler   = distributed.DistributedSampler(
        val_ds,   num_replicas=world, rank=rank, shuffle=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)
    val_dl   = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=False)

    # -------------- Model -------------- #
    model = Base_model(
        in_nc=args.in_nc, out_nc=args.out_nc,
        nf=args.nf, gc=args.ns, times=args.times,
        normalize=args.normalize).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # -------------- Optim / Scheduler -------------- #
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler  = ReduceLROnPlateau(optimizer, mode="min",
                                   factor=args.scheduler_factor,
                                   patience=args.scheduler_patience,
                                   min_lr=args.scheduler_min_lr,
                                   verbose=(rank==0))

    # -------------- Trainer (reuse) -------------- #
    # trainingmodule.Trainer 는 single-GPU 전제이므로,
    # rank 0 프로세스에서만 checkpoint / wandb 를 기록하도록 래퍼 작성
    trainer = trainingmodule.Trainer(
        train_dl, val_dl, model, optimizer, scheduler, opt=args)
    best_val = math.inf
    no_impr  = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)      # ★ DDP epoch-shuffle
        trainer.train(epoch)                # 내부에서 model.train()

        # --- validation (각 rank 계산 → 평균) --- #
        with torch.inference_mode():
            val_loss, _ = trainer.validation(epoch)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        average_all(val_loss_tensor, world)   # 모든 rank 평균
        mean_val_loss = val_loss_tensor.item()

        if rank == 0:
            logger.info(f"[E{epoch:03d}] val_loss={mean_val_loss:.4f}")
            wandb.log({"val_loss": mean_val_loss}, step=epoch)

            # checkpoint (only master)
            if mean_val_loss < best_val:
                best_val = mean_val_loss
                os.makedirs(args.save_folder, exist_ok=True)
                torch.save(model.module.state_dict(),
                           f"{args.save_folder}/best_val.pth")
        scheduler.step(mean_val_loss)

    if rank == 0: wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
