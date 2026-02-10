"""Visualization helpers for stereo/disparity outputs."""

from __future__ import annotations

import numpy as np


ArrayF = np.ndarray


def make_anaglyph(left: ArrayF, right: ArrayF) -> ArrayF:
    """Build a red-cyan anaglyph from two RGB images."""
    l = np.asarray(left, dtype=np.float32)
    r = np.asarray(right, dtype=np.float32)
    if l.shape != r.shape:
        raise ValueError("left and right must have the same shape")
    if l.ndim != 3 or l.shape[-1] != 3:
        raise ValueError("anaglyph expects HxWx3 images")

    out = np.empty_like(l)
    out[..., 0] = l[..., 0]
    out[..., 1] = r[..., 1]
    out[..., 2] = r[..., 2]
    return np.clip(out, 0.0, 1.0)


def normalize_disparity(d: ArrayF, vmin: float, vmax: float) -> ArrayF:
    """Normalize disparity to [0, 1] using shared min/max."""
    disp = np.asarray(d, dtype=np.float32)
    denom = max(vmax - vmin, 1e-6)
    return np.clip((disp - vmin) / denom, 0.0, 1.0)


def disparity_to_rgb(d_norm: ArrayF) -> ArrayF:
    """Map normalized disparity to a simple perceptual-ish RGB ramp."""
    t = np.asarray(d_norm, dtype=np.float32)
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.6 * t - 0.1, 0.0, 1.0)
    g = np.clip(1.0 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
    b = np.clip(1.2 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def amplified_abs_diff(reference: ArrayF, estimate: ArrayF, gain: float = 6.0) -> ArrayF:
    """Visualize absolute difference with amplification."""
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")

    diff = np.abs(ref - est)
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    diff = np.clip(gain * diff, 0.0, 1.0)
    return np.stack([diff, diff, diff], axis=-1)
