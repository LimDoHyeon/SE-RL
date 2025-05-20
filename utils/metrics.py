"""Common evaluation metrics wrappers (CPU‑only external libs)."""
from typing import Union
import torch
import numpy as np
from pesq import pesq  # Itu‑T P.862 implementation (pip install pesq)

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
        scores.append(pesq(sr, _to_numpy(c), _to_numpy(d), "wb"))
    return torch.tensor(scores).mean()