"""Disparity remap operators."""

from __future__ import annotations

import numpy as np


ArrayF = np.ndarray


def scale_shift(d: ArrayF, scale: float, shift: float) -> ArrayF:
    """Scale and shift disparity values."""
    disp = np.asarray(d, dtype=np.float32)
    return disp * np.float32(scale) + np.float32(shift)


def soft_clip(d: ArrayF, min_d: float, max_d: float, softness: float) -> ArrayF:
    """Softly clip disparity into [min_d, max_d] using tanh compression."""
    disp = np.asarray(d, dtype=np.float32)
    if softness <= 0.0:
        return np.clip(disp, min_d, max_d)

    center = 0.5 * (min_d + max_d)
    half_range = 0.5 * (max_d - min_d)
    if half_range <= 0.0:
        return np.full_like(disp, fill_value=min_d, dtype=np.float32)

    normalized = (disp - center) / (softness + 1e-6)
    mapped = np.tanh(normalized)
    return center + half_range * mapped


def object_popout(d: ArrayF, mask: np.ndarray, delta: float) -> ArrayF:
    """Increase disparity by delta inside a binary object mask."""
    disp = np.asarray(d, dtype=np.float32)
    out = disp.copy()
    out[np.asarray(mask, dtype=bool)] += np.float32(delta)
    return out


def depth_grade(d: ArrayF, grade_field: ArrayF, strength: float) -> ArrayF:
    """Apply a spatially varying depth grade field."""
    disp = np.asarray(d, dtype=np.float32)
    field = np.asarray(grade_field, dtype=np.float32)
    return disp + np.float32(strength) * field
