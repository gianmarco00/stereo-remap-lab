from __future__ import annotations

from stereo_remap.metrics import psnr
from stereo_remap.synthetic import SlantedPlaneScene
from stereo_remap.warp.backward import backward_valid_mask_from_disparity, backward_warp_monotonic


def test_backward_warp_matches_slanted_plane_ground_truth() -> None:
    scene = SlantedPlaneScene()
    left, disparity, right_gt = scene.render()
    right = backward_warp_monotonic(left, disparity)
    valid = backward_valid_mask_from_disparity(disparity) & scene.right_valid_mask()
    assert psnr(right_gt, right, mask=valid) > 45.0
