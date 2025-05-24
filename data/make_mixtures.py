#!/usr/bin/env python3
"""
make_mixtures.py
================
Offline data-generation script for SERL project (VoiceBank-DEMAND, mono 16 kHz).

It creates noisy–clean segment pairs with 50 % overlap (16 384 samples) at
defined SNR grids, following the experimental protocol in the reference paper.

Usage example
-------------
python make_mixtures.py \
    --clean_dir /data/VoiceBank/clean \
    --noise_dir /data/DEMAND/wav \
    --split_dir ./splits \
    --out_root ./vb_demand_seg \
    --workers 16 \
    --seed 42
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# -----------------------------------------------------------------------------#
#                             Fixed experiment spec                             #
# -----------------------------------------------------------------------------#
SPLIT_NAMES = ["train0", "train1", "test0", "test1"]

SNR_TABLE = {
    "train0": [15, 10, 5, 0],
    "train1": [15, 10, 5, 0],
    "test0":  [15, 10, 5, 0],
    "test1":  [17.5, 12.5, 7.5, 2.5],   # mismatch SNR set
}

# *Example* DEMAND category lists – adjust as needed
NOISE_LIST = {
    "train0": ["airport", "babble", "car", "restaurant", "street",
               "subway", "metro", "office", "park", "station"],
    "train1": ["airport", "babble", "car", "restaurant", "street",
               "subway", "metro", "office", "park", "station"],
    "test0":  ["airport", "babble", "car", "restaurant", "street",
               "subway", "metro", "office", "park", "station"],
    "test1":  ["airport", "babble", "car", "restaurant", "street"],  # 5 types
}

SEG_LEN = 16_384
HOP_LEN = SEG_LEN // 2   # 50 % overlap
SR = 16_000
PEAK_CLIP = 0.99


# -----------------------------------------------------------------------------#
#                               Core utilities                                  #
# -----------------------------------------------------------------------------#
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor**2))


def choose_noise_file(noise_root: Path, noise_cat: str) -> Path:
    """Randomly pick one noise wav from a category sub-directory."""
    path = noise_root / noise_cat
    candidates = list(path.glob("*.wav"))
    assert candidates, f"No noise wavs in {path}"
    return random.choice(candidates)


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    assert sr == SR, f"SR mismatch {sr}"
    if wav.dim() > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)   # mono
    return wav


def mix_at_snr(clean: torch.Tensor,
               noise: torch.Tensor,
               snr_db: float) -> torch.Tensor:
    """Scale noise to achieve target SNR, then add."""
    # noise length adjust
    if noise.size(1) < clean.size(1):
        rep = math.ceil(clean.size(1) / noise.size(1))
        noise = noise.repeat(1, rep)[:, : clean.size(1)]
    else:
        start = random.randint(0, noise.size(1) - clean.size(1))
        noise = noise[:, start : start + clean.size(1)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)
    snr_linear = 10 ** (snr_db / 20)
    noise = noise * clean_rms / (snr_linear * noise_rms)

    noisy = clean + noise
    peak = noisy.abs().max()
    if peak > PEAK_CLIP:
        noisy = noisy / peak * PEAK_CLIP
        clean = clean / peak * PEAK_CLIP
    return noisy, clean


def segment_and_save(noisy: torch.Tensor,
                     clean: torch.Tensor,
                     speaker: str,
                     base_name: str,
                     out_noisy_dir: Path,
                     out_clean_dir: Path) -> None:
    total = noisy.size(1)
    idx = 0
    for st in range(0, total - SEG_LEN + 1, HOP_LEN):
        en = st + SEG_LEN
        noisy_seg = noisy[:, st:en]
        clean_seg = clean[:, st:en]
        fname = f"{base_name}_{idx:03d}.wav"
        torchaudio.save(str(out_noisy_dir / speaker / fname), noisy_seg, SR, encoding="PCM_S", bits_per_sample=16)
        torchaudio.save(str(out_clean_dir / speaker / fname), clean_seg, SR, encoding="PCM_S", bits_per_sample=16)
        idx += 1
    # tail < SEG_LEN/2 is discarded per spec


# -----------------------------------------------------------------------------#
#                              Worker procedure                                 #
# -----------------------------------------------------------------------------#
def process_utterance(args: Tuple[str, str, str, Path, Path, Path]):
    split, clean_rel, speaker_id, clean_root, noise_root, out_root = args

    # deterministic randomness per utterance
    base_seed = hash(clean_rel) & 0xFFFF_FFFF
    set_global_seed(base_seed)

    clean_path = clean_root / clean_rel
    clean = load_wav(clean_path)

    noise_cat = random.choice(NOISE_LIST[split])
    noise_path = choose_noise_file(noise_root, noise_cat)
    noise = load_wav(noise_path)

    snr = random.choice(SNR_TABLE[split])
    noisy, clean_scaled = mix_at_snr(clean, noise, snr)

    # prepare speaker sub-dirs once per process
    out_noisy_dir = out_root / split / "noisy" / speaker_id
    out_clean_dir = out_root / split / "clean" / speaker_id
    out_noisy_dir.mkdir(parents=True, exist_ok=True)
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(clean_rel).stem  # e.g. p226_001
    segment_and_save(noisy, clean_scaled,
                     speaker_id, base_name,
                     out_noisy_dir, out_clean_dir)
    return 1  # for tally


# -----------------------------------------------------------------------------#
#                                   Main                                        #
# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--clean_dir", required=True, help="Root of clean wavs (VoiceBank corpus)")
    parser.add_argument("--noise_dir", required=True, help="Root of DEMAND wavs")
    parser.add_argument("--split_dir", required=True, help="Directory containing train0.lst train1.lst ...")
    parser.add_argument("--out_root", required=True, help="Output root directory for generated segments")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    args = parser.parse_args()

    set_global_seed(args.seed)
    clean_root = Path(args.clean_dir).expanduser()
    noise_root = Path(args.noise_dir).expanduser()
    out_root = Path(args.out_root).expanduser()

    tasks = []
    for split in SPLIT_NAMES:
        list_path = Path(args.split_dir) / f"{split}.lst"
        assert list_path.exists(), f"{list_path} not found"

        with open(list_path) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                spk = rel.split("/")[0] if "/" in rel else rel.split("_")[0]
                tasks.append((split, rel, spk, clean_root, noise_root, out_root))

    with mp.Pool(processes=args.workers, initializer=set_global_seed, initargs=(args.seed,)) as pool:
        for _ in tqdm(pool.imap_unordered(process_utterance, tasks),
                      total=len(tasks), desc="Mixing"):
            pass

    # summary
    print(f"Completed. Mixtures saved under: {out_root}")


if __name__ == "__main__":
    main()
