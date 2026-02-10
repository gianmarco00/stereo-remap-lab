"""Geometry and sampling utilities used by stereo warpers."""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray


def right_coords_from_disparity(disparity_row: ArrayF) -> ArrayF:
    """Compute right-view x coordinates from a left row disparity map."""
    width = int(disparity_row.shape[0])
    x_left = np.arange(width, dtype=np.float32)
    return x_left - disparity_row.astype(np.float32)


def is_monotonic_non_decreasing(values: ArrayF, atol: float = 1e-6) -> bool:
    """Check if a 1D array is non-decreasing within a tolerance."""
    deltas = np.diff(values.astype(np.float64), axis=0)
    return bool(np.all(deltas >= -atol))


def invert_monotonic_mapping(
    x_from: ArrayF,
    x_to: ArrayF,
    query: ArrayF,
) -> ArrayF:
    """Invert a monotonic 1D mapping with linear interpolation and edge clamping."""
    if not is_monotonic_non_decreasing(x_to):
        raise ValueError("x_to must be monotonic non-decreasing for inversion")
    return np.interp(query, x_to, x_from, left=x_from[0], right=x_from[-1]).astype(np.float32)


def sample_row_bilinear(row: ArrayF, x_coords: ArrayF) -> ArrayF:
    """Sample a row (W,C) or (W,) at floating x coordinates with bilinear interpolation."""
    row_arr = np.asarray(row, dtype=np.float32)
    squeeze = row_arr.ndim == 1
    if squeeze:
        row_arr = row_arr[:, None]

    width = row_arr.shape[0]
    x = np.clip(np.asarray(x_coords, dtype=np.float32), 0.0, float(width - 1))
    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    t = (x - x0).astype(np.float32)[:, None]

    sampled = (1.0 - t) * row_arr[x0] + t * row_arr[x1]
    if squeeze:
        return sampled[:, 0]
    return sampled
