"""Visualization helpers for stereo/disparity outputs."""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray


def make_anaglyph(left: ArrayF, right: ArrayF) -> ArrayF:
    """Build a red-cyan anaglyph from two RGB images."""
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right must have the same shape")
    if left_arr.ndim != 3 or left_arr.shape[-1] != 3:
        raise ValueError("anaglyph expects HxWx3 images")

    out = np.empty_like(left_arr)
    out[..., 0] = left_arr[..., 0]
    out[..., 1] = right_arr[..., 1]
    out[..., 2] = right_arr[..., 2]
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


def mask_to_rgb(mask: np.ndarray) -> ArrayF:
    """Convert a boolean mask into a black/white RGB image."""
    m = np.asarray(mask, dtype=bool).astype(np.float32)
    return np.stack([m, m, m], axis=-1)


def apply_invalid_overlay(
    image: ArrayF,
    valid_mask: np.ndarray,
    invalid_color: tuple[float, float, float] = (0.07, 0.07, 0.07),
) -> ArrayF:
    """Paint invalid pixels with a neutral overlay color."""
    img = np.asarray(image, dtype=np.float32).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    if img.shape[:2] != valid.shape:
        raise ValueError("valid_mask must match image height and width")
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("image must be HxWx3")
    img[~valid] = np.asarray(invalid_color, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)
