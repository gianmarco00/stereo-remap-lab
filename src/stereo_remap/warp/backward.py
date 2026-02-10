"""Backward warping for monotonic left->right mappings."""

from __future__ import annotations

import numpy as np

from stereo_remap.geometry import (
    invert_monotonic_mapping,
    right_coords_from_disparity,
    sample_row_bilinear,
)

ArrayF = np.ndarray


def backward_valid_mask_from_disparity(disparity: ArrayF) -> np.ndarray:
    """Compute valid right-view pixels implied by monotonic left->right mapping."""
    disp = np.asarray(disparity, dtype=np.float32)
    if disp.ndim != 2:
        raise ValueError("disparity must be HxW")

    height, width = disp.shape
    x_right = np.arange(width, dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)

    for y in range(height):
        xr_from_xl = right_coords_from_disparity(disp[y])
        lo = float(min(xr_from_xl[0], xr_from_xl[-1]))
        hi = float(max(xr_from_xl[0], xr_from_xl[-1]))
        valid[y] = (x_right >= lo) & (x_right <= hi)
    return valid


def backward_warp_monotonic(left: ArrayF, disparity: ArrayF) -> ArrayF:
    """Warp LEFT image to RIGHT using per-row inversion of x_r(x_l) = x_l - d(x_l)."""
    left_arr = np.asarray(left, dtype=np.float32)
    disp = np.asarray(disparity, dtype=np.float32)

    if left_arr.ndim not in (2, 3):
        raise ValueError("left must be HxW or HxWxC")
    if disp.shape != left_arr.shape[:2]:
        raise ValueError("disparity must have shape HxW matching left image")

    squeeze = left_arr.ndim == 2
    if squeeze:
        left_arr = left_arr[:, :, None]

    height, width, channels = left_arr.shape
    x_left = np.arange(width, dtype=np.float32)
    x_right_query = np.arange(width, dtype=np.float32)
    out = np.empty_like(left_arr, dtype=np.float32)

    for y in range(height):
        xr_from_xl = right_coords_from_disparity(disp[y])
        x_left_for_right = invert_monotonic_mapping(x_left, xr_from_xl, x_right_query)
        out[y] = sample_row_bilinear(left_arr[y], x_left_for_right)

    if squeeze:
        return out[:, :, 0]
    return out
