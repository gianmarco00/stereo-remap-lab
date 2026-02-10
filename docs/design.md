# Stereo Remap Lab Design

## Goals
- Keep stereo convention explicit and consistent: `d = x_left - x_right`.
- Provide deterministic, testable warping behavior in pure NumPy.
- Demonstrate both clean monotonic warping and general forward splatting with holes.

## Scene strategy
- `SlantedPlaneScene` uses affine-in-x disparity so each row is monotonic and invertible.
- `LayeredRectScene` uses a near foreground rectangle over a far background to force disocclusions.

## Warpers
- Backward monotonic warp inverts `x_r(x_l)` per row via linear interpolation and samples the left row bilinearly.
- Forward z-buffer warp splats each source pixel to nearest integer target and keeps the largest disparity.

## Hole filling
- Rowwise nearest fill quickly removes NaNs.
- Diffusion smoothing improves visual continuity inside previously missing regions.

## Determinism
- Synthetic scenes are parameterized with fixed seeds and no nondeterministic IO.
- Outputs are stable across runs for tests and CI.
