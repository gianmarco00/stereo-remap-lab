from __future__ import annotations

import numpy as np

from stereo_remap.metrics import l1_error, psnr


def test_masked_metrics_ignore_invalid_regions() -> None:
    ref = np.zeros((2, 3, 1), dtype=np.float32)
    est = ref.copy()
    est[0, 0, 0] = 1.0
    mask = np.array([[False, True, True], [True, True, True]], dtype=bool)

    assert np.isinf(psnr(ref, est, mask=mask))
    assert l1_error(ref, est, mask=mask) == 0.0
