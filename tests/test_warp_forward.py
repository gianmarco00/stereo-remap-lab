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
