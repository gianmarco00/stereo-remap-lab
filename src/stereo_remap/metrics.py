"""Image quality metrics."""

from __future__ import annotations

import math

import numpy as np


ArrayF = np.ndarray


def psnr(reference: ArrayF, estimate: ArrayF, data_range: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio in dB."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    mse = np.mean((ref - est) ** 2)
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def l1_error(reference: ArrayF, estimate: ArrayF) -> float:
    """Compute mean absolute photometric error."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    return float(np.mean(np.abs(ref - est)))
