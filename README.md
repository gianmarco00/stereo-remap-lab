# Stereo Remap Lab (Portfolio Edition)

A compact, deterministic stereo/disparity remapping lab in pure Python (`numpy` + `pillow`).

## Stereo convention
- Disparity: `d = x_left - x_right`
- Left-to-right synthesis: `x_right = x_left - d`
- Larger positive disparity means closer depth.

## Features
- Deterministic synthetic scenes:
  - `SlantedPlaneScene` (monotonic, analytic right-view ground truth)
  - `LayeredRectScene` (occlusions/disocclusions)
- Two warpers:
  - Backward monotonic row inversion + bilinear sampling
  - Forward z-buffer splat with hole mask
- Hole filling:
  - Rowwise nearest fill
  - Diffusion smoothing inside holes
- Disparity remap operators:
  - `scale_shift`, `soft_clip`, `object_popout`, `depth_grade`
- Metrics:
  - `PSNR`, `L1` photometric error

## Install
```bash
pip install -e ".[dev]"
```

## Run
```bash
pytest -q
python scripts/make_showcase.py
python scripts/benchmark.py
```

`make_showcase.py` writes outputs to `artifacts/showcase/` and copies committed preview assets to `docs/assets/`.

## Repository layout
```text
stereo-remap-lab/
  README.md
  LICENSE
  pyproject.toml
  .github/workflows/ci.yml
  AGENTS.md
  src/stereo_remap/
    __init__.py
    io.py
    geometry.py
    metrics.py
    pipeline.py
    synthetic.py
    viz.py
    remap/operators.py
    warp/forward.py
    warp/backward.py
    warp/holes.py
  scripts/
    make_showcase.py
    benchmark.py
  tests/
    test_geometry.py
    test_warp_backward.py
    test_warp_forward.py
    test_remap_ops.py
    test_pipeline_smoke.py
  docs/
    assets/
    design.md
```

## Showcase
### Inputs/outputs
![Left view](docs/assets/left.png)
![Right GT](docs/assets/right_gt.png)
![Right after remap](docs/assets/right_after.png)

### Stereo fusion
![Anaglyph before](docs/assets/anaglyph_before.png)
![Anaglyph after](docs/assets/anaglyph_after.png)

### Disparity and error
![Disparity before](docs/assets/disp_before.png)
![Disparity after](docs/assets/disp_after.png)
![Amplified absolute error](docs/assets/diff_gt_vs_after.png)
