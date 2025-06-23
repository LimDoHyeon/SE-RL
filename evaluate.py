#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SERL – Evaluation script (stable composite metrics)
Author: Dohyeon

• Distortion metrics  : RMSE, SegSNR, SI‑SDR
• Perceptual metrics  : PESQ, CSIG, CBAK (via speechmetrics GitHub 0.4.x)
• Test set            : Valentini Test‑1 (list: test1.lst)
• GPU                 : DataParallel (2 GPUs expected)
• Batch size          : 16
• Output              : Console log + CSV
• UX                  : tqdm progress bar per‑batch
"""
from __future__ import annotations

import argparse
import csv
import logging
import os, sys
from typing import Dict, List
from pathlib import Path

import torch, torchaudio
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.dataloader import AudioDataset, pad_collate, worker_init_fn
from model.modules2 import DynamicFilterModel
from utils.metrics import rmse, segsnr, sisdr_batch, composite_wrap, pesq_wrapper
from utils.util import update_namespace_from_yaml


def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s ▸ %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("SERL‑Eval")


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sr: int,
    log: logging.Logger,
    out_dir: Path,
    gen_wav: int,
):
    """Iterates over `loader`, returns (avg_dict, rows_list)."""

    model.eval()
    rows: List[Dict[str, float]] = []
    saved = 0

    for noisy, clean, rels in tqdm(loader, desc="Evaluating", unit="batch",
                                   ascii=True, dynamic_ncols=True, file=sys.stdout):
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy, deterministic=True)
        enh = out[1] if isinstance(out, (list, tuple)) and len(out) >= 2 else out if not isinstance(out,(list, tuple)) else out[0]

        # metrics ---------------------------------------------------------- #
        row = {
            "rmse":   rmse(clean, enh).item(),
            "segSNR": segsnr(clean, enh, sr).item(),
            "siSDR":  sisdr_batch(enh, clean).item(),
            "pesq":   pesq_wrapper(clean.squeeze(1), enh.squeeze(1), sr).item(),
        }
        # csig, cbak = composite_wrap(clean, enh, sr)
        # row.update({"csig": csig.item(), "cbak": cbak.item()})
        rows.append(row)

    # waveform saving --------------------------------------------- #
    if gen_wav > 0 and saved < gen_wav:
        batch_size = noisy.size(0)
        for b in range(batch_size):
            if saved >= gen_wav:
                break
            rel_path = rels[b]
            # ensure Path type
            rel_path = rel_path if isinstance(rel_path, str) else rel_path.decode()
            enh_path = out_dir / f"enh_{rel_path}"
            noisy_path_out = out_dir / f"noisy_{rel_path}"
            enh_path.parent.mkdir(parents=True, exist_ok=True)
            noisy_path_out.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(enh_path.as_posix(), enh[b].cpu(), sr)
            torchaudio.save(noisy_path_out.as_posix(), noisy[b].cpu(), sr)
            saved += 1

    # aggregate
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    log.info(
        # "▸ RMSE %.4f | SegSNR %.2f dB | SI‑SDR %.2f dB | PESQ %.3f | CSIG %.3f | CBAK %.3f",
        # avg["rmse"], avg["segSNR"], avg["siSDR"], avg["pesq"], avg["csig"], avg["cbak"]
        "▸ RMSE %.4f | SegSNR %.2f dB | SI‑SDR %.2f dB | PESQ %.3f",
        avg["rmse"], avg["segSNR"], avg["siSDR"], avg["pesq"]
    )
    return avg, rows


def main(*, config_file: str, noisy_root: str, clean_root: str, list_file: str,
         checkpoint: str, out_dir: str, generate_waveform: int):

    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    log = _logger()
    args = update_namespace_from_yaml(argparse.Namespace(), config_file)
    sr = args.sample_rate

    # dataset -------------------------------------------------------------- #
    ds = AudioDataset(
        noisy_root=noisy_root,
        clean_root=clean_root,
        list_file=list_file,
        sample_rate=sr,
        segment_len=None,
        random_crop=False,
        dataset_length=1 << 60,
        subset_seed=args.subset_seed,
    )
    dl = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
        worker_init_fn=worker_init_fn,
    )

    # model --------------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicFilterModel(
        in_nc=args.in_nc,
        c=args.c,
        w=args.w,
        Ns=args.Ns,
        Nf=args.Nf,
        gc=args.gc,
        nr=args.nr,
        m=args.m,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if torch.cuda.device_count() > 1:
        log.info("Using %d GPUs (DataParallel)", torch.cuda.device_count())
        model = DataParallel(model)

    # evaluate ------------------------------------------------------------- #
    avg, rows = _evaluate(model, dl, device, sr, log, out_dir_p, generate_waveform)

    # save metrics CSV -------------------------------------------------- #
    csv_path = out_dir_p / "metrics.csv"
    # keys = ["rmse", "segSNR", "siSDR", "pesq", "csig", "cbak"]
    keys = ["rmse", "segSNR", "siSDR", "pesq"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows); wr.writerow({k: avg[k] for k in keys})
    log.info("Metrics CSV saved → %s", csv_path)


if __name__ == "__main__":
    if __name__ == "__main__":
        p = argparse.ArgumentParser("SERL evaluator – paths & options")
        p.add_argument("--config", "-c", default="config/train_config.yaml")
        p.add_argument("--noisy_root", default="data/Valentini/valentini")
        p.add_argument("--clean_root", default="data/Valentini/valentini")
        p.add_argument("--list_file", default="data/Valentini/valentini/test1.lst")
        p.add_argument("--checkpoint", "-ckpt", default="exp/serl-bs32/nnet_iter113_trloss0.0250_valoss0.0244.pt")
        p.add_argument("--out_dir", "-o", default="exp/eval_bs32/eval_out_113")
        p.add_argument("--generate_waveform", "-gen", type=int, default=0,
                       help="save first N samples (noisy & enhanced) as wav")
        args = p.parse_args()

        main(
            config_file=args.config,
            noisy_root=args.noisy_root,
            clean_root=args.clean_root,
            list_file=args.list_file,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            generate_waveform=args.generate_waveform,
        )
