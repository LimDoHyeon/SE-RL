"""Common evaluation metrics wrappers (CPU‑only external libs)."""
from typing import Union
import torch
import numpy as np
from pesq import pesq, NoUtterancesError  # Itu‑T P.862 implementation (pip install pesq)

__all__ = ["pesq_wrapper"]


def _to_numpy(x: Union[torch.Tensor, np.ndarray]):
    """Ensure input is 1‑D NumPy array on CPU, float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return x.astype(np.float32)


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