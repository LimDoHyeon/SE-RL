
# src/data_loader/dataloader.py
"""
AudioDataset & helper utilities for SERL project
------------------------------------------------
* Pure **data-loading** -- mixing / SNR augmentation / overlap segmentation are
  assumed to be handled by an external *data-generation* pipeline.
* Returns a (noisy, clean) **pair of tensors** shaped  [1, T]  (float32, −1∼1).
* Segment-level handling (fixed‐length crop) is optional; if ``segment_len`` is
  None the whole file is returned and the collate-fn zero-pads to the max-len
  in the mini-batch.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

__all__ = [
    "AudioDataset",
    "pad_collate",
    "worker_init_fn",
]

# -------------------- Helper functions -------------------- #
def set_deterministic_seed(seed: int) -> None:
    """Seed python / numpy / torch for DataLoader worker reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init_fn(worker_id: int) -> None:
    """Each DataLoader worker gets a different, but deterministic, RNG seed."""
    base_seed = torch.initial_seed() % 2**32
    set_deterministic_seed(base_seed + worker_id)


def pad_collate(batch):
    noisy, clean, rels = zip(*batch)            # 3-튜플 언패킹
    lengths = torch.tensor([x.size(1) for x in noisy], dtype=torch.long)

    max_len = int(lengths.max())
    padded_noisy = torch.zeros(len(noisy), 1, max_len, dtype=noisy[0].dtype)
    padded_clean = torch.zeros_like(padded_noisy)

    for i, (n, c) in enumerate(zip(noisy, clean)):
        T = n.size(1)
        padded_noisy[i, :, :T] = n
        padded_clean[i, :, :T] = c

    return padded_noisy, padded_clean, rels      # ← lengths 대신 rels 반환


# -------------------- Dataset -------------------- #
class AudioDataset(Dataset):
    """
    Parameters
    ----------
    noisy_root : str
        Directory that contains **noisy** audio files produced by the
        data-generation pipeline.
    clean_root : str
        Directory that contains matching **clean** reference audio files.
    list_file : str
        Text file - each line is either:
          1) ``<utt>.wav``                (noisy/clean roots share structure) or
          2) ``<noisy_rel_path> <clean_rel_path>`` .
    segment_len : Optional[int], default ``16384`` (≈1 s @16 kHz)
        If set, the dataset returns fixed-length segments.  If *None* the whole
        file is returned.
    random_crop : bool, default ``True``
        When ``segment_len`` is not None, choose the crop start randomly
        (training); if **False**, start at 0 (validation/test).
    scale_aug : bool, default ``False``
        Simple amplitude scaling augmentation ±20 %.
    lowpass_aug : bool, default ``False``
        Random 6-kHz or 8-kHz low-pass filter as mild bandwidth reduction.
    sample_rate : int, default ``16000``
        Expected sample rate.  An assertion is raised if files differ.
    """

    def __init__(
        self,
        noisy_root: str,
        clean_root: str,
        list_file: str,
        *,
        segment_len: Optional[int] = None,
        random_crop: bool = True,
        scale_aug: bool = False,
        lowpass_aug: bool = False,
        sample_rate: int = 16000,
        dataset_length: Optional[int] = None,
        subset_seed: Optional[int] = None,          # ── NEW
        noisy_substr: str = "/noisy/",
        clean_substr: str = "/clean/",
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Public hyper-parameters
        # ------------------------------------------------------------------ #
        self.noisy_root       = noisy_root
        self.clean_root       = clean_root
        self.segment_len      = segment_len
        self.random_crop      = random_crop
        self.scale_aug        = scale_aug
        self.lowpass_aug      = lowpass_aug
        self.sample_rate      = sample_rate
        self.dataset_length   = dataset_length
        self.subset_seed      = subset_seed          # ── NEW

        # ---------------- read full file list ---------------- #
        self._full_pairs: List[Tuple[str, str]] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                toks = ln.strip().split()
                if len(toks) == 1:
                    noisy_rel = toks[0]
                    clean_rel = noisy_rel.replace(noisy_substr, clean_substr, 1)
                    if noisy_rel == clean_rel:
                        raise ValueError(
                            f"Auto-inferred clean path == noisy path for line: {ln.strip()}"
                        )
                elif len(toks) >= 2:
                    noisy_rel, clean_rel = toks[0], toks[1]
                else:
                    raise ValueError(f"Malformed line in list file: {ln}")
                self._full_pairs.append((noisy_rel, clean_rel))

        # -------- deterministic random subset selection ------ #
        self._select_subset()                       # ── NEW

    # ------------------------------------------------------------------ #
    # Internal subset selector
    # ------------------------------------------------------------------ #
    def _select_subset(self) -> None:               # ── NEW
        """Populate ``self.file_pairs`` according to seed / length policy."""
        if self.dataset_length is None or self.dataset_length >= len(self._full_pairs):
            # keep all
            self.file_pairs = list(self._full_pairs)
            return

        if self.subset_seed is None:
            # Legacy behaviour: pick *first* N examples
            self.file_pairs = self._full_pairs[: self.dataset_length]   # :contentReference[oaicite:0]{index=0}
        else:
            # Deterministic random sampling w/ independent RNG
            rng = random.Random(self.subset_seed)
            idx = rng.sample(range(len(self._full_pairs)), k=self.dataset_length)
            self.file_pairs = [self._full_pairs[i] for i in idx]

    # ------------------------------------------------------------------ #
    # Optional: reseed per-epoch
    # ------------------------------------------------------------------ #
    def reseed_subset(self, epoch: int) -> None:    # ── NEW
        if self.subset_seed is None or self.dataset_length is None:
            return  # nothing to do
        new_seed = self.subset_seed + epoch
        rng = random.Random(new_seed)
        idx = rng.sample(range(len(self._full_pairs)), k=self.dataset_length)
        self.file_pairs = [self._full_pairs[i] for i in idx]

    # ------------------------------------------- #
    def __len__(self) -> int:
        return len(self.file_pairs)

    # ------------------------------------------- #
    @staticmethod
    def _load_audio(path: str, expected_sr: int) -> torch.Tensor:
        """
        Returns
        -------
        Tensor shaped [1, num_frames], float32, −1 … 1
        """
        wav, sr = torchaudio.load(path)
        assert sr == expected_sr, f"Sample rate mismatch: {sr} vs {expected_sr}"
        if wav.dim() == 2 and wav.size(0) > 1:
            # ignore second channel and use first channel only
            wav = wav[:1, :]
        return wav

    # ------------------------------------------- #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_rel, clean_rel = self.file_pairs[idx]
        noisy_path = os.path.join(self.noisy_root, noisy_rel)
        clean_path = os.path.join(self.clean_root, clean_rel)
        noisy, sr = torchaudio.load(noisy_path)
        clean, _ = torchaudio.load(clean_path)

        return noisy, clean, noisy_rel  # ← rel_path 추가