from __future__ import annotations

import numpy as np

from stereo_remap.remap.operators import depth_grade, object_popout, scale_shift, soft_clip


def test_scale_shift_mean_is_expected() -> None:
    disp = np.linspace(1.0, 5.0, 100, dtype=np.float32).reshape(10, 10)
    scale = 1.3
    shift = 0.7
    out = scale_shift(disp, scale=scale, shift=shift)

    expected = scale * float(disp.mean()) + shift
    assert abs(float(out.mean()) - expected) < 1e-6


def test_other_remap_operators_basic_properties() -> None:
    disp = np.array([[0.0, 1.0], [2.0, 10.0]], dtype=np.float32)
    clipped = soft_clip(disp, min_d=1.0, max_d=6.0, softness=0.75)
    assert clipped.min() >= 1.0 - 1e-3
    assert clipped.max() <= 6.0 + 1e-3

    mask = np.array([[False, True], [False, True]])
    popped = object_popout(disp, mask, delta=2.0)
    np.testing.assert_allclose(popped[mask], disp[mask] + 2.0)

    field = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    graded = depth_grade(disp, field, strength=0.5)
    np.testing.assert_allclose(graded, disp + 0.5 * field, atol=1e-6)
