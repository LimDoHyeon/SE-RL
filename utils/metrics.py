"""Common evaluation metrics wrappers (CPU‑only external libs)."""
from __future__ import annotations
from typing import Tuple, Union

import torch
import numpy as np
from pesq import pesq, NoUtterancesError                 # ITU-T P.862
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as sisdr,
)
from speechmetrics import load as _load_speechmetrics     # CSIG / CBAK

__all__ = [
    "rmse",
    "segsnr",
    "sisdr_batch",
    "composite_wrap",
    "pesq_wrapper",
]


def _to_numpy(x: Union[torch.Tensor, np.ndarray]):
    """Ensure input is 1‑D NumPy array on CPU, float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return x.astype(np.float32)

# ----------------------------- distortion metrics --------------------------------
def rmse(clean: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
    """Root-MSE over batch."""
    return torch.sqrt(torch.mean((clean - est) ** 2, dim=[1, 2])).mean()


def segsnr(
    clean: torch.Tensor,
    est: torch.Tensor,
    sr: int = 16_000,
    frame_len: float = 0.03,
) -> torch.Tensor:
    """Segmental SNR (ITU-T P.56 스타일, 30 ms window, 50 % hop)."""
    B, _, T = clean.shape
    frame = int(frame_len * sr)
    hop = frame // 2
    seg_scores = []
    for b in range(B):
        s = 0
        scores = []
        while s + frame <= T:
            c_seg = clean[b, 0, s : s + frame]
            e_seg = est[b, 0, s : s + frame]
            sig = torch.sum(c_seg ** 2) + 1e-8
            noi = torch.sum((c_seg - e_seg) ** 2) + 1e-8
            scores.append(10 * torch.log10(sig / noi))
            s += hop
        seg_scores.append(torch.stack(scores).mean())
    return torch.stack(seg_scores).mean()


def sisdr_batch(est: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """ Batch-mean SI-SDR (torchmetrics) - performance metrics only """
    if est.dim() == 3:  # [B, 1, T] → [B, T]
        est, clean = est.squeeze(1), clean.squeeze(1)
    return sisdr(est, clean).mean()


# ----------------------------- perceptual metrics --------------------------------
def pesq_wrapper(clean: torch.Tensor, degraded: torch.Tensor, sr: int = 16000):
    """Batch‑wise PESQ (wide‑band) – returns mean PESQ over batch.

    Args:
        clean:  [B, T] clean reference (tensor or ndarray)
        degraded: [B, T] noisy / enhanced audio
        sr: sampling rate (default 16 kHz)
    Returns:
        float Tensor – mean PESQ score (on current device)
    """
    assert clean.shape == degraded.shape, "clean & degraded must match in shape"
    scores = []
    for c, d in zip(clean, degraded):
        ref = _to_numpy(c)
        deg = _to_numpy(d)
        try:
            score = pesq(sr, ref, deg, 'wb')
        except NoUtterancesError:
            score = 0.0
        scores.append(score)
    return torch.tensor(scores).mean()

_speechmetrics_cache: dict[int, any] = {}

def composite_wrap(
    clean: torch.Tensor, est: torch.Tensor, sr: int = 16_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (csig_mean, cbak_mean) as torch scalars.
    Uses speechmetrics 'composite' package.
    """
    if sr not in _speechmetrics_cache:
        _speechmetrics_cache[sr] = _load_speechmetrics("composite", window=2)
    comp = _speechmetrics_cache[sr]

    c_np = clean.squeeze(1).cpu().numpy()
    e_np = est.squeeze(1).cpu().numpy()
    out = comp(e_np, c_np)  # {'csig': …, 'cbak': …}

    return (torch.tensor(float(out["csig"])),
            torch.tensor(float(out["cbak"])))
