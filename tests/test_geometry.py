from __future__ import annotations

import numpy as np

from stereo_remap.geometry import (
    invert_monotonic_mapping,
    is_monotonic_non_decreasing,
    right_coords_from_disparity,
    sample_row_bilinear,
)


def test_right_coords_and_monotonicity() -> None:
    disparity = np.array([1.0, 1.2, 1.4, 1.8], dtype=np.float32)
    xr = right_coords_from_disparity(disparity)
    np.testing.assert_allclose(
        xr,
        np.array([-1.0, -0.2, 0.6, 1.2], dtype=np.float32),
        atol=1e-6,
    )
    assert is_monotonic_non_decreasing(xr)


def test_invert_monotonic_mapping_and_sampling() -> None:
    x_from = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x_to = np.array([0.0, 0.5, 2.0, 3.5], dtype=np.float32)
    query = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    x_inv = invert_monotonic_mapping(x_from, x_to, query)

    row = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    sampled = sample_row_bilinear(row, x_inv)

    np.testing.assert_allclose(
        x_inv,
        np.array([0.0, 1.3333334, 2.6666667], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sampled,
        np.array([0.0, 13.333334, 26.666668], dtype=np.float32),
        atol=1e-4,
    )
