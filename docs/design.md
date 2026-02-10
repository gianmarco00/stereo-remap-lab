# Stereo Remap Lab Design

## Goals
- Keep stereo convention explicit and consistent: `d = x_left - x_right`.
- Provide deterministic, testable warping behavior in pure NumPy.
- Demonstrate both clean monotonic warping and general forward splatting with holes.

## Scene strategy
- `SlantedPlaneScene` uses affine-in-x disparity so each row is monotonic and invertible.
- The scene texture is intentionally human-readable (sky, horizon, buildings, road markers) so shifts are interpretable without specialist context.
- `LayeredRectScene` uses a near foreground rectangle over a far background to force disocclusions.
- The foreground rectangle is rendered as a sign-like panel with a hard border to make popout behavior obvious.

## Warpers
- Backward monotonic warp inverts `x_r(x_l)` per row via linear interpolation and samples the left row bilinearly.
- Forward z-buffer warp splats each source pixel to nearest integer target and keeps the largest disparity.
- Backward valid masks are computed analytically and used for fair masked metrics and visual overlays.

## Hole filling
- Rowwise nearest fill quickly removes NaNs.
- Diffusion smoothing improves visual continuity inside previously missing regions.
- Diffusion uses edge padding (not circular wrapping), so top/bottom and left/right borders never bleed into each other.

## Determinism
- Synthetic scenes are parameterized with fixed seeds and no nondeterministic IO.
- Outputs are stable across runs for tests and CI.
- Showcase images crop to a shared valid field-of-view for fair visual comparison while metrics use explicit masks.
