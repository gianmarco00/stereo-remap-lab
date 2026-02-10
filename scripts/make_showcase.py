#!/usr/bin/env python3
"""Generate deterministic showcase artifacts for disparity remapping."""

from __future__ import annotations

import sys
from pathlib import Path
import shutil

import numpy as np

# Allow running directly from a clone without editable install.
if __package__ is None or __package__ == "":
    _SRC_DIR = Path(__file__).resolve().parents[1] / "src"
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

from stereo_remap.io import ensure_dir, write_image
from stereo_remap.metrics import l1_error, psnr
from stereo_remap.pipeline import run_slanted_plane_pipeline
from stereo_remap.viz import (
    amplified_abs_diff,
    disparity_to_rgb,
    make_anaglyph,
    normalize_disparity,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts" / "showcase"
    docs_assets_dir = root / "docs" / "assets"
    ensure_dir(out_dir)
    ensure_dir(docs_assets_dir)

    result = run_slanted_plane_pipeline(scale=1.35, shift=0.75)

    dmin = float(np.min([result.disp_before.min(), result.disp_after.min()]))
    dmax = float(np.max([result.disp_before.max(), result.disp_after.max()]))

    disp_before_rgb = disparity_to_rgb(normalize_disparity(result.disp_before, dmin, dmax))
    disp_after_rgb = disparity_to_rgb(normalize_disparity(result.disp_after, dmin, dmax))

    anaglyph_before = make_anaglyph(result.left, result.right_before)
    anaglyph_after = make_anaglyph(result.left, result.right_after)
    diff_amp = amplified_abs_diff(result.right_gt, result.right_after, gain=8.0)

    outputs = {
        "left.png": result.left,
        "right_gt.png": result.right_gt,
        "right_after.png": result.right_after,
        "anaglyph_before.png": anaglyph_before,
        "anaglyph_after.png": anaglyph_after,
        "disp_before.png": disp_before_rgb,
        "disp_after.png": disp_after_rgb,
        "diff_gt_vs_after.png": diff_amp,
    }

    for name, img in outputs.items():
        write_image(out_dir / name, img)

    selected = [
        "left.png",
        "right_gt.png",
        "right_after.png",
        "anaglyph_before.png",
        "anaglyph_after.png",
        "disp_before.png",
        "disp_after.png",
        "diff_gt_vs_after.png",
    ]
    for name in selected:
        shutil.copy2(out_dir / name, docs_assets_dir / name)

    before_psnr = psnr(result.right_gt, result.right_before)
    after_psnr = psnr(result.right_gt, result.right_after)
    after_l1 = l1_error(result.right_gt, result.right_after)

    print(f"Showcase written to: {out_dir}")
    print(f"Copied images to:  {docs_assets_dir}")
    print(f"PSNR(right_gt, right_before): {before_psnr:.3f} dB")
    print(f"PSNR(right_gt, right_after):  {after_psnr:.3f} dB")
    print(f"L1(right_gt, right_after):    {after_l1:.6f}")


if __name__ == "__main__":
    main()
