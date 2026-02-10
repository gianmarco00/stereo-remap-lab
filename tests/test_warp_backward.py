from __future__ import annotations

from stereo_remap.metrics import psnr
from stereo_remap.synthetic import SlantedPlaneScene
from stereo_remap.warp.backward import backward_warp_monotonic


def test_backward_warp_matches_slanted_plane_ground_truth() -> None:
    left, disparity, right_gt = SlantedPlaneScene().render()
    right = backward_warp_monotonic(left, disparity)
    assert psnr(right_gt, right) > 45.0
