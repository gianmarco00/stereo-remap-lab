"""High-level pipelines used by scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereo_remap.remap.operators import object_popout, scale_shift
from stereo_remap.synthetic import LayeredRectScene, SlantedPlaneScene
from stereo_remap.warp.backward import backward_valid_mask_from_disparity, backward_warp_monotonic
from stereo_remap.warp.forward import forward_zbuffer_warp
from stereo_remap.warp.holes import diffusion_fill_holes, rowwise_nearest_fill

ArrayF = np.ndarray


@dataclass(frozen=True)
class SlantedPlanePipelineResult:
    left: ArrayF
    right_gt: ArrayF
    right_before: ArrayF
    right_after: ArrayF
    disp_before: ArrayF
    disp_after: ArrayF
    valid_before: np.ndarray
    valid_after: np.ndarray
    valid_gt: np.ndarray


@dataclass(frozen=True)
class LayeredRectPipelineResult:
    left: ArrayF
    right_before_nan: ArrayF
    right_before: ArrayF
    right_after_nan: ArrayF
    right_after: ArrayF
    disp_before: ArrayF
    disp_after: ArrayF
    hole_mask_before: np.ndarray
    hole_mask_after: np.ndarray


def run_slanted_plane_pipeline(
    scale: float = 1.20,
    shift: float = 0.50,
) -> SlantedPlanePipelineResult:
    """Run monotonic slanted-plane remap showcase pipeline."""
    scene = SlantedPlaneScene()
    left, disp_before, right_gt = scene.render()

    right_before = backward_warp_monotonic(left, disp_before)
    disp_after = scale_shift(disp_before, scale=scale, shift=shift)
    right_after = backward_warp_monotonic(left, disp_after)
    valid_before = backward_valid_mask_from_disparity(disp_before)
    valid_after = backward_valid_mask_from_disparity(disp_after)

    return SlantedPlanePipelineResult(
        left=left,
        right_gt=right_gt,
        right_before=right_before,
        right_after=right_after,
        disp_before=disp_before,
        disp_after=disp_after,
        valid_before=valid_before,
        valid_after=valid_after,
        valid_gt=scene.right_valid_mask(),
    )


def run_layered_rect_pipeline(
    popout_delta: float = 2.50,
) -> LayeredRectPipelineResult:
    """Run occlusion-heavy layered-rectangle pipeline with hole filling."""
    scene = LayeredRectScene()
    left, disp_before, _ = scene.render()

    right_before_nan, hole_before = forward_zbuffer_warp(left, disp_before)
    right_before = diffusion_fill_holes(
        rowwise_nearest_fill(right_before_nan),
        hole_before,
        iterations=16,
    )

    disp_after = object_popout(disp_before, scene.foreground_mask(), delta=popout_delta)
    right_after_nan, hole_after = forward_zbuffer_warp(left, disp_after)
    right_after = diffusion_fill_holes(
        rowwise_nearest_fill(right_after_nan),
        hole_after,
        iterations=16,
    )

    return LayeredRectPipelineResult(
        left=left,
        right_before_nan=right_before_nan,
        right_before=right_before,
        right_after_nan=right_after_nan,
        right_after=right_after,
        disp_before=disp_before,
        disp_after=disp_after,
        hole_mask_before=hole_before,
        hole_mask_after=hole_after,
    )
