"""Forward left->right warp with per-pixel z-buffer splatting."""

from __future__ import annotations

import numpy as np


ArrayF = np.ndarray


def forward_zbuffer_warp(left: ArrayF, disparity: ArrayF) -> tuple[ArrayF, np.ndarray]:
    """Forward-splat LEFT to RIGHT and keep largest disparity on collisions."""
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
    warped = np.full((height, width, channels), np.nan, dtype=np.float32)
    zbuf = np.full((height, width), -np.inf, dtype=np.float32)

    for y in range(height):
        for x_left in range(width):
            d = float(disp[y, x_left])
            x_right = int(np.rint(x_left - d))
            if x_right < 0 or x_right >= width:
                continue
            if d >= zbuf[y, x_right]:
                zbuf[y, x_right] = d
                warped[y, x_right, :] = left_arr[y, x_left, :]

    hole_mask = np.isnan(warped).any(axis=-1)
    if squeeze:
        return warped[:, :, 0], hole_mask
    return warped, hole_mask
