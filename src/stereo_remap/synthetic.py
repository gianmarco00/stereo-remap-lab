"""Deterministic synthetic scenes for stereo remapping experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereo_remap.geometry import sample_row_bilinear


ArrayF = np.ndarray


def _texture_rgb(x: ArrayF, y: ArrayF, seed: int) -> ArrayF:
    """Evaluate a deterministic textured RGB pattern at continuous coordinates."""
    rng = np.random.default_rng(seed)
    p0, p1, p2 = rng.uniform(0.0, 2.0 * np.pi, size=3)

    bars = (np.floor((x + 0.35 * y) / 12.0) % 2.0).astype(np.float32)
    checker = (np.floor(x / 24.0) + np.floor(y / 20.0)) % 2.0

    r = 0.12 + 0.52 * bars + 0.24 * np.sin(0.10 * x + 0.05 * y + p0)
    g = 0.16 + 0.45 * checker + 0.25 * np.cos(0.07 * x - 0.08 * y + p1)
    b = 0.18 + 0.34 * bars + 0.22 * checker + 0.22 * np.sin(0.12 * y + p2)

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


@dataclass(frozen=True)
class SlantedPlaneScene:
    """Monotonic slanted-plane disparity scene with analytic right-view synthesis."""

    width: int = 384
    height: int = 216
    seed: int = 7
    slope_x: float = 0.045
    base_disp: float = 2.0
    slope_y: float = 1.5

    def render(self) -> tuple[ArrayF, ArrayF, ArrayF]:
        """Return (left, disparity, right_gt) as float32 arrays in [0, 1] and pixels."""
        x = np.arange(self.width, dtype=np.float32)[None, :]
        y = np.arange(self.height, dtype=np.float32)[:, None]
        y_norm = y / max(float(self.height - 1), 1.0)

        disparity = self.slope_x * x + self.base_disp + self.slope_y * y_norm
        disparity = disparity.astype(np.float32)

        xx = np.broadcast_to(x, (self.height, self.width))
        yy = np.broadcast_to(y, (self.height, self.width))
        left = _texture_rgb(xx, yy, seed=self.seed)

        # Closed-form inverse for x_r = x_l - d(x_l, y), where d is affine in x_l.
        x_right = xx
        b_row = self.base_disp + self.slope_y * y_norm
        x_left_from_right = (x_right + b_row) / max(1.0 - self.slope_x, 1e-6)
        x_left_from_right = np.clip(x_left_from_right, 0.0, float(self.width - 1))

        right_gt = np.empty_like(left, dtype=np.float32)
        for row_idx in range(self.height):
            right_gt[row_idx] = sample_row_bilinear(left[row_idx], x_left_from_right[row_idx])

        return left.astype(np.float32), disparity.astype(np.float32), right_gt.astype(np.float32)


@dataclass(frozen=True)
class LayeredRectScene:
    """Background plus foreground rectangle with larger disparity that induces disocclusions."""

    width: int = 384
    height: int = 216
    seed: int = 13
    bg_disparity: float = 2.0
    fg_disparity: float = 14.0

    def foreground_mask(self) -> np.ndarray:
        """Return a deterministic foreground rectangle mask."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        x0 = self.width // 4
        x1 = (self.width * 3) // 5
        y0 = self.height // 5
        y1 = (self.height * 4) // 5
        mask[y0:y1, x0:x1] = True
        return mask

    def render(self) -> tuple[ArrayF, ArrayF, ArrayF]:
        """Return (left, disparity, right_hint) for layered-occlusion demos."""
        x = np.arange(self.width, dtype=np.float32)[None, :]
        y = np.arange(self.height, dtype=np.float32)[:, None]
        xx = np.broadcast_to(x, (self.height, self.width))
        yy = np.broadcast_to(y, (self.height, self.width))

        left = _texture_rgb(xx, yy, seed=self.seed)
        mask = self.foreground_mask()

        # Make the foreground object visibly distinct with hard edges.
        fg_color = np.array([0.95, 0.30, 0.22], dtype=np.float32)
        left = left.copy()
        left[mask] = 0.7 * left[mask] + 0.3 * fg_color

        disparity = np.full((self.height, self.width), self.bg_disparity, dtype=np.float32)
        disparity[mask] = self.fg_disparity

        return left.astype(np.float32), disparity.astype(np.float32), left.astype(np.float32)
