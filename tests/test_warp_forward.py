from __future__ import annotations

import numpy as np

from stereo_remap.synthetic import LayeredRectScene
from stereo_remap.warp.forward import forward_zbuffer_warp
from stereo_remap.warp.holes import diffusion_fill_holes, rowwise_nearest_fill


def test_forward_warp_produces_holes_on_layered_scene() -> None:
    left, disparity, _ = LayeredRectScene().render()
    warped, hole_mask = forward_zbuffer_warp(left, disparity)

    assert hole_mask.any()
    assert np.isnan(warped).any()


def test_hole_filling_leaves_no_nans() -> None:
    left, disparity, _ = LayeredRectScene().render()
    warped, hole_mask = forward_zbuffer_warp(left, disparity)

    filled = rowwise_nearest_fill(warped)
    smooth = diffusion_fill_holes(filled, hole_mask, iterations=12)

    assert not np.isnan(filled).any()
    assert not np.isnan(smooth).any()


def test_diffusion_fill_does_not_wrap_image_borders() -> None:
    image = np.zeros((4, 4, 1), dtype=np.float32)
    image[0, :, 0] = 0.25
    image[-1, :, 0] = 1.0
    hole_mask = np.zeros((4, 4), dtype=bool)
    hole_mask[0, 1:3] = True
    image[hole_mask, 0] = np.nan

    filled = rowwise_nearest_fill(image)
    smoothed = diffusion_fill_holes(filled, hole_mask, iterations=8)

    assert float(smoothed[0, 1, 0]) < 0.5
    assert float(smoothed[0, 2, 0]) < 0.5
