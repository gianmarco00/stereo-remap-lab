from __future__ import annotations

import numpy as np

from stereo_remap.pipeline import run_layered_rect_pipeline, run_slanted_plane_pipeline


def test_pipeline_smoke_runs_both_scenes() -> None:
    slanted = run_slanted_plane_pipeline()
    layered = run_layered_rect_pipeline()

    assert slanted.left.shape == slanted.right_gt.shape
    assert slanted.left.shape == slanted.right_after.shape
    assert np.isfinite(slanted.disp_before).all()
    assert np.isfinite(slanted.disp_after).all()
    assert slanted.valid_before.any()
    assert slanted.valid_after.any()

    assert layered.left.shape == layered.right_before.shape
    assert layered.left.shape == layered.right_after.shape
    assert layered.hole_mask_before.any()
    assert layered.hole_mask_after.any()
    assert np.isnan(layered.right_before_nan).any()
    assert np.isnan(layered.right_after_nan).any()
    assert np.isfinite(layered.right_before).all()
    assert np.isfinite(layered.right_after).all()
