"""Hole filling operators for forward-warp outputs."""

from __future__ import annotations

import numpy as np


ArrayF = np.ndarray


def _fill_row_nearest_valid(row: ArrayF) -> ArrayF:
    """Fill NaNs in one row using nearest valid sample in x."""
    out = np.asarray(row, dtype=np.float32).copy()
    if out.ndim == 1:
        out = out[:, None]
        squeeze = True
    else:
        squeeze = False

    width = out.shape[0]
    x = np.arange(width)

    valid_any = ~np.isnan(out).any(axis=-1)
    if not np.any(valid_any):
        out[:] = 0.0
        return out[:, 0] if squeeze else out

    valid_idx = x[valid_any]
    query_idx = x[~valid_any]
    insert_pos = np.searchsorted(valid_idx, query_idx, side="left")
    left_pos = np.clip(insert_pos - 1, 0, valid_idx.size - 1)
    right_pos = np.clip(insert_pos, 0, valid_idx.size - 1)
    left_idx = valid_idx[left_pos]
    right_idx = valid_idx[right_pos]
    choose_right = np.abs(query_idx - right_idx) < np.abs(query_idx - left_idx)
    nearest_idx = left_idx.copy()
    nearest_idx[choose_right] = right_idx[choose_right]
    out[~valid_any] = out[nearest_idx]

    if squeeze:
        return out[:, 0]
    return out


def rowwise_nearest_fill(image_with_nans: ArrayF) -> ArrayF:
    """Fill NaN holes independently in each row with nearest horizontal samples."""
    arr = np.asarray(image_with_nans, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError("image_with_nans must be HxW or HxWxC")

    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[:, :, None]

    out = arr.copy()
    for y in range(out.shape[0]):
        out[y] = _fill_row_nearest_valid(out[y])

    if squeeze:
        return out[:, :, 0]
    return out


def diffusion_fill_holes(image: ArrayF, hole_mask: np.ndarray, iterations: int = 20) -> ArrayF:
    """Diffuse colors inside holes while preserving known pixels."""
    arr = np.asarray(image, dtype=np.float32)
    holes = np.asarray(hole_mask, dtype=bool)

    if arr.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    if holes.shape != arr.shape[:2]:
        raise ValueError("hole_mask must match first two dimensions of image")

    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[:, :, None]

    out = arr.copy()
    out[np.isnan(out)] = 0.0

    for _ in range(max(iterations, 0)):
        up = np.roll(out, shift=1, axis=0)
        down = np.roll(out, shift=-1, axis=0)
        left = np.roll(out, shift=1, axis=1)
        right = np.roll(out, shift=-1, axis=1)
        avg = 0.25 * (up + down + left + right)
        out[holes] = avg[holes]

    if squeeze:
        return out[:, :, 0]
    return out
