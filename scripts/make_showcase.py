#!/usr/bin/env python3
"""Generate deterministic showcase artifacts for disparity remapping."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

# Allow running directly from a clone without editable install.
if __package__ is None or __package__ == "":
    _SRC_DIR = Path(__file__).resolve().parents[1] / "src"
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

from stereo_remap.io import ensure_dir, write_image
from stereo_remap.metrics import l1_error, psnr
from stereo_remap.pipeline import run_layered_rect_pipeline, run_slanted_plane_pipeline
from stereo_remap.viz import (
    amplified_abs_diff,
    apply_invalid_overlay,
    disparity_to_rgb,
    make_anaglyph,
    mask_to_rgb,
    normalize_disparity,
)


def _common_valid_slice(valid_mask: np.ndarray) -> slice:
    """Return the widest column slice that is valid for every row."""
    cols = np.where(np.all(valid_mask, axis=0))[0]
    if cols.size == 0:
        return slice(0, valid_mask.shape[1])
    return slice(int(cols[0]), int(cols[-1]) + 1)


def _crop_columns(image: np.ndarray, col_slice: slice) -> np.ndarray:
    """Crop an image-like array on width axis only."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[:, col_slice]
    return arr[:, col_slice, :]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts" / "showcase"
    docs_assets_dir = root / "docs" / "assets"
    ensure_dir(out_dir)
    ensure_dir(docs_assets_dir)

    result = run_slanted_plane_pipeline(scale=1.35, shift=0.75)
    layered = run_layered_rect_pipeline(popout_delta=2.5)

    dmin = float(np.min([result.disp_before.min(), result.disp_after.min()]))
    dmax = float(np.max([result.disp_before.max(), result.disp_after.max()]))

    disp_before_rgb = disparity_to_rgb(normalize_disparity(result.disp_before, dmin, dmax))
    disp_after_rgb = disparity_to_rgb(normalize_disparity(result.disp_after, dmin, dmax))

    valid_before = result.valid_before & result.valid_gt
    valid_after = result.valid_after & result.valid_gt
    valid_intersection = valid_before & valid_after
    slanted_cols = _common_valid_slice(valid_intersection)

    left_vis = _crop_columns(result.left, slanted_cols)
    right_gt_vis = _crop_columns(result.right_gt, slanted_cols)
    right_before_vis = _crop_columns(result.right_before, slanted_cols)
    right_after_vis = _crop_columns(result.right_after, slanted_cols)
    disp_before_vis = _crop_columns(disp_before_rgb, slanted_cols)
    disp_after_vis = _crop_columns(disp_after_rgb, slanted_cols)

    anaglyph_before = make_anaglyph(left_vis, right_before_vis)
    anaglyph_after = make_anaglyph(left_vis, right_after_vis)
    diff_amp = amplified_abs_diff(right_gt_vis, right_after_vis, gain=8.0)

    outputs = {
        "left.png": left_vis,
        "right_gt.png": right_gt_vis,
        "right_after.png": right_after_vis,
        "right_before.png": right_before_vis,
        "valid_gt.png": mask_to_rgb(result.valid_gt),
        "valid_after.png": mask_to_rgb(valid_after),
        "anaglyph_before.png": anaglyph_before,
        "anaglyph_after.png": anaglyph_after,
        "disp_before.png": disp_before_vis,
        "disp_after.png": disp_after_vis,
        "diff_gt_vs_after.png": diff_amp,
        "layered_left.png": layered.left,
        "layered_holes_before.png": mask_to_rgb(layered.hole_mask_before),
        "layered_holes_after.png": mask_to_rgb(layered.hole_mask_after),
        "layered_right_before_nans.png": apply_invalid_overlay(
            np.nan_to_num(layered.right_before_nan, nan=0.0),
            ~layered.hole_mask_before,
            invalid_color=(1.0, 0.0, 1.0),
        ),
        "layered_right_before_filled.png": layered.right_before,
        "layered_right_after_nans.png": apply_invalid_overlay(
            np.nan_to_num(layered.right_after_nan, nan=0.0),
            ~layered.hole_mask_after,
            invalid_color=(1.0, 0.0, 1.0),
        ),
        "layered_right_after_filled.png": layered.right_after,
    }

    for name, img in outputs.items():
        write_image(out_dir / name, img)

    selected = [
        "left.png",
        "right_gt.png",
        "right_before.png",
        "right_after.png",
        "valid_gt.png",
        "valid_after.png",
        "anaglyph_before.png",
        "anaglyph_after.png",
        "disp_before.png",
        "disp_after.png",
        "diff_gt_vs_after.png",
        "layered_left.png",
        "layered_holes_before.png",
        "layered_holes_after.png",
        "layered_right_before_nans.png",
        "layered_right_before_filled.png",
        "layered_right_after_nans.png",
        "layered_right_after_filled.png",
    ]
    for name in selected:
        shutil.copy2(out_dir / name, docs_assets_dir / name)

    before_psnr = psnr(result.right_gt, result.right_before, mask=valid_before)
    after_psnr = psnr(result.right_gt, result.right_after, mask=valid_intersection)
    after_l1 = l1_error(result.right_gt, result.right_after, mask=valid_intersection)

    print(f"Showcase written to: {out_dir}")
    print(f"Copied images to:  {docs_assets_dir}")
    print(f"PSNR(right_gt, right_before): {before_psnr:.3f} dB")
    print(f"PSNR(right_gt, right_after):  {after_psnr:.3f} dB")
    print(f"L1(right_gt, right_after):    {after_l1:.6f}")
    print(
        "Layered holes: "
        f"before={float(layered.hole_mask_before.mean()):.3f}, "
        f"after={float(layered.hole_mask_after.mean()):.3f}"
    )


if __name__ == "__main__":
    main()
