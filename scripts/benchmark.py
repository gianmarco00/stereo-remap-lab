#!/usr/bin/env python3
"""Simple CPU benchmark for remap operators and warpers."""

from __future__ import annotations

import argparse
import time

from stereo_remap.pipeline import run_layered_rect_pipeline, run_slanted_plane_pipeline
from stereo_remap.synthetic import LayeredRectScene, SlantedPlaneScene
from stereo_remap.warp.backward import backward_warp_monotonic
from stereo_remap.warp.forward import forward_zbuffer_warp
from stereo_remap.warp.holes import diffusion_fill_holes, rowwise_nearest_fill


def _timeit(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / max(iterations, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark stereo remap operators")
    parser.add_argument("--iters", type=int, default=20, help="Iterations per operation")
    args = parser.parse_args()

    left_s, disp_s, _ = SlantedPlaneScene().render()
    left_l, disp_l, _ = LayeredRectScene().render()

    t_backward = _timeit(lambda: backward_warp_monotonic(left_s, disp_s), args.iters)

    def _forward_and_fill() -> None:
        warped, holes = forward_zbuffer_warp(left_l, disp_l)
        filled = rowwise_nearest_fill(warped)
        diffusion_fill_holes(filled, holes, iterations=12)

    t_forward = _timeit(_forward_and_fill, args.iters)
    t_pipeline_slanted = _timeit(run_slanted_plane_pipeline, max(1, args.iters // 2))
    t_pipeline_layered = _timeit(run_layered_rect_pipeline, max(1, args.iters // 2))

    print("Stereo Remap Benchmark (ms per run)")
    print(f"backward_warp_monotonic : {t_backward:8.3f}")
    print(f"forward+fill pipeline    : {t_forward:8.3f}")
    print(f"slanted pipeline         : {t_pipeline_slanted:8.3f}")
    print(f"layered pipeline         : {t_pipeline_layered:8.3f}")


if __name__ == "__main__":
    main()
