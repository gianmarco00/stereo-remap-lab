"""Image quality metrics."""

from __future__ import annotations

import math

import numpy as np

ArrayF = np.ndarray


def _masked_values(
    reference: ArrayF,
    estimate: ArrayF,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select masked elements for metric computation."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")

    if mask is None:
        return ref.reshape(-1), est.reshape(-1)

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != ref.shape[:2]:
        raise ValueError("mask must match first two dimensions of input arrays")
    if not np.any(mask_arr):
        raise ValueError("mask must include at least one valid pixel")

    if ref.ndim == 3:
        return ref[mask_arr], est[mask_arr]
    else:
        return ref[mask_arr], est[mask_arr]


def psnr(
    reference: ArrayF,
    estimate: ArrayF,
    data_range: float = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Compute peak signal-to-noise ratio in dB."""
    ref, est = _masked_values(reference, estimate, mask)
    mse = np.mean((ref - est) ** 2)
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def l1_error(
    reference: ArrayF,
    estimate: ArrayF,
    mask: np.ndarray | None = None,
) -> float:
    """Compute mean absolute photometric error."""
    ref, est = _masked_values(reference, estimate, mask)
    return float(np.mean(np.abs(ref - est)))
