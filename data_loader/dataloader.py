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
from torch.utils.data import Dataset, DataLoader


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


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function that zero-pads variable-length waveforms to the length of
    the longest sample in the mini-batch.

    Returns
    -------
    noisy_batch : Tensor  [B, 1, T_max]
    clean_batch : Tensor  [B, 1, T_max]
    lengths     : Tensor  [B]      valid lengths before padding
    """
    noisy, clean = zip(*batch)                        # tuple of tensors
    lengths = torch.tensor([x.size(1) for x in noisy], dtype=torch.long)

    max_len = int(lengths.max())
    padded_noisy = torch.zeros(len(noisy), 1, max_len, dtype=noisy[0].dtype)
    padded_clean = torch.zeros_like(padded_noisy)

    for i, (n, c) in enumerate(zip(noisy, clean)):
        T = n.size(1)
        padded_noisy[i, :, :T] = n
        padded_clean[i, :, :T] = c

    return padded_noisy, padded_clean, lengths


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
        noisy_root: str, clean_root: str, list_file: str,
        segment_len: Optional[int] = 16384, random_crop: bool = True,
        scale_aug: bool = False, lowpass_aug: bool = False,
        sample_rate: int = 16000,):

        super().__init__()

        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.segment_len = segment_len
        self.random_crop = random_crop
        self.scale_aug = scale_aug
        self.lowpass_aug = lowpass_aug
        self.sample_rate = sample_rate

        # ------------ read file list ------------ #
        with open(list_file, "r", encoding="utf-8") as f:
            raw = [ln.strip() for ln in f if ln.strip()]

        self.file_pairs: List[Tuple[str, str]] = []
        for line in raw:
            fields = line.split()
            if len(fields) == 1:
                utt = fields[0]
                self.file_pairs.append((utt, utt))
            elif len(fields) == 2:
                self.file_pairs.append(tuple(fields))
            else:
                raise ValueError(f"Invalid list line: {line}")

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
    def _maybe_crop(self, wav: torch.Tensor) -> torch.Tensor:
        if self.segment_len is None:
            return wav

        total = wav.size(1)
        if total < self.segment_len:
            # zero-pad short utterances
            padded = torch.zeros(1, self.segment_len, dtype=wav.dtype)
            padded[:, :total] = wav
            return padded

        if self.random_crop:
            max_start = total - self.segment_len
            start = random.randint(0, max_start)
        else:
            start = 0  # deterministic

        end = start + self.segment_len
        return wav[:, start:end]

    # ------------------------------------------- #
    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        pass

    # ------------------------------------------- #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_rel, clean_rel = self.file_pairs[idx]
        noisy_path = os.path.join(self.noisy_root, noisy_rel)
        clean_path = os.path.join(self.clean_root, clean_rel)

        noisy = self._load_audio(noisy_path, self.sample_rate)
        clean = self._load_audio(clean_path, self.sample_rate)

        # optional cropping
        # noisy = self._maybe_crop(noisy)
        # clean = self._maybe_crop(clean)

        # light augmentation (noisy only)
        # noisy = self._augment(noisy)

        assert noisy.shape == clean.shape, "Shape mismatch after cropping."
        return noisy.contiguous(), clean.contiguous()


# -----------------------------------------------------------------------------#
#                           Public factory convenience                         #
# -----------------------------------------------------------------------------#
def build_dataloader(
    noisy_root: str, clean_root: str, list_file: str,
    batch_size: int, segment_len: Optional[int] = 16384,
    shuffle: bool = True, num_workers: int = 4,
    random_crop: bool = False, scale_aug: bool = False,
    lowpass_aug: bool = False, sample_rate: int = 16000,
    pin_memory: bool = True,) -> DataLoader:
    """
    Simplified helper that instantiates ``AudioDataset`` and wraps it in a
    ``torch.utils.data.DataLoader``.
    """
    dataset = AudioDataset(
        noisy_root=noisy_root, clean_root=clean_root, list_file=list_file,
        segment_len=segment_len, random_crop=random_crop,
        scale_aug=scale_aug, lowpass_aug=lowpass_aug,
        sample_rate=sample_rate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate if segment_len is None else None,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataloader
