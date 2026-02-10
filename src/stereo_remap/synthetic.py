"""Deterministic synthetic scenes for stereo remapping experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereo_remap.geometry import sample_row_bilinear

ArrayF = np.ndarray


def _texture_rgb(x: ArrayF, y: ArrayF, seed: int) -> ArrayF:
    """Evaluate a deterministic scene texture with human-readable structure."""
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    max_x = max(float(np.max(x_arr)), 1.0)
    max_y = max(float(np.max(y_arr)), 1.0)
    x_norm = x_arr / max_x
    y_norm = y_arr / max_y

    horizon = 0.56
    sky = np.stack(
        [
            0.58 - 0.26 * y_norm + 0.04 * np.sin(5.5 * x_norm),
            0.72 - 0.28 * y_norm + 0.03 * np.sin(4.1 * x_norm + 0.8),
            0.90 - 0.20 * y_norm,
        ],
        axis=-1,
    )
    ground = np.stack(
        [
            0.28 + 0.26 * y_norm,
            0.28 + 0.22 * y_norm,
            0.30 + 0.19 * y_norm,
        ],
        axis=-1,
    )
    blend = np.clip((y_norm - (horizon - 0.05)) / 0.10, 0.0, 1.0)
    texture = (1.0 - blend[..., None]) * sky + blend[..., None] * ground

    road_mask = y_norm > (horizon + 0.02)
    road_markers = (np.floor((x_arr + 0.75 * y_arr) / 24.0) % 2.0) == 0.0
    texture[road_mask & road_markers] = np.array([0.78, 0.78, 0.74], dtype=np.float32)
    curb = np.abs(y_norm - horizon) < (1.2 / max_y)
    texture[curb] = np.array([0.93, 0.93, 0.90], dtype=np.float32)

    rng = np.random.default_rng(seed)
    building_count = 6
    for idx in range(building_count):
        x0 = 0.03 + idx * 0.16 + float(rng.uniform(-0.01, 0.01))
        width = 0.10 + float(rng.uniform(-0.015, 0.03))
        x1 = min(x0 + width, 0.98)
        top = 0.20 + float(rng.uniform(-0.03, 0.05))

        building = (
            (x_norm >= x0)
            & (x_norm < x1)
            & (y_norm >= top)
            & (y_norm < horizon)
        )
        shade = 0.35 + 0.07 * idx
        color = np.array([shade * 0.92, shade * 0.95, shade], dtype=np.float32)
        texture[building] = color

        window_x = (np.floor((x_norm - x0) * max_x / 14.0) % 2.0) == 0.0
        window_y = (np.floor((y_norm - top) * max_y / 10.0) % 2.0) == 0.0
        windows = building & window_x & window_y
        texture[windows] = np.array([0.92, 0.86, 0.68], dtype=np.float32)

    return np.clip(texture.astype(np.float32), 0.0, 1.0)


@dataclass(frozen=True)
class SlantedPlaneScene:
    """Monotonic slanted-plane disparity scene with analytic right-view synthesis."""

    width: int = 384
    height: int = 216
    seed: int = 7
    slope_x: float = 0.045
    base_disp: float = 2.0
    slope_y: float = 1.5

    def right_valid_mask(self) -> np.ndarray:
        """Return pixels in RIGHT view with in-bounds source coordinates."""
        x = np.arange(self.width, dtype=np.float32)[None, :]
        y = np.arange(self.height, dtype=np.float32)[:, None]
        y_norm = y / max(float(self.height - 1), 1.0)
        b_row = self.base_disp + self.slope_y * y_norm
        x_left_from_right = (x + b_row) / max(1.0 - self.slope_x, 1e-6)
        return (x_left_from_right >= 0.0) & (x_left_from_right <= float(self.width - 1))

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

        valid = self.right_valid_mask()
        right_gt = np.zeros_like(left, dtype=np.float32)
        for row_idx in range(self.height):
            row = sample_row_bilinear(left[row_idx], x_left_from_right[row_idx])
            right_gt[row_idx, valid[row_idx]] = row[valid[row_idx]]

        return left.astype(np.float32), disparity.astype(np.float32), right_gt.astype(np.float32)


@dataclass(frozen=True)
class LayeredRectScene:
    """Background plus foreground rectangle with larger disparity that induces disocclusions."""

    width: int = 384
    height: int = 216
    seed: int = 13
    bg_disparity: float = 2.0
    fg_disparity: float = 14.0

    def foreground_bounds(self) -> tuple[int, int, int, int]:
        """Return rectangle bounds (x0, x1, y0, y1) for the foreground object."""
        x0 = self.width // 4
        x1 = (self.width * 3) // 5
        y0 = self.height // 5
        y1 = (self.height * 4) // 5
        return x0, x1, y0, y1

    def foreground_mask(self) -> np.ndarray:
        """Return a deterministic foreground rectangle mask."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        x0, x1, y0, y1 = self.foreground_bounds()
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

        left = left.copy()
        x0, x1, y0, y1 = self.foreground_bounds()
        obj_x = (xx - float(x0)) / max(float(x1 - x0 - 1), 1.0)
        obj_y = (yy - float(y0)) / max(float(y1 - y0 - 1), 1.0)

        panel = np.stack(
            [
                0.18 + 0.10 * np.sin(6.0 * obj_x) + 0.03 * obj_y,
                0.40 + 0.09 * np.cos(5.0 * obj_y),
                0.76 - 0.14 * obj_y + 0.03 * np.sin(4.0 * obj_x),
            ],
            axis=-1,
        )
        badge = (
            (obj_x > 0.18)
            & (obj_x < 0.82)
            & (obj_y > 0.18)
            & (obj_y < 0.44)
        )
        panel[badge] = np.array([0.96, 0.88, 0.28], dtype=np.float32)
        text_rows = badge & ((np.floor((obj_y - 0.18) * 48.0) % 2.0) == 0.0)
        panel[text_rows] = np.array([0.26, 0.22, 0.10], dtype=np.float32)
        left[mask] = np.clip(panel[mask], 0.0, 1.0)

        border = mask & (
            (xx <= float(x0 + 1))
            | (xx >= float(x1 - 2))
            | (yy <= float(y0 + 1))
            | (yy >= float(y1 - 2))
        )
        left[border] = np.array([0.98, 0.98, 0.96], dtype=np.float32)

        shadow = (
            (xx >= float(x0 + 5))
            & (xx <= float(x1 + 7))
            & (yy >= float(y1))
            & (yy <= float(min(y1 + 8, self.height - 1)))
        )
        left[shadow & ~mask] *= 0.72

        disparity = np.full((self.height, self.width), self.bg_disparity, dtype=np.float32)
        disparity[mask] = self.fg_disparity

        return left.astype(np.float32), disparity.astype(np.float32), left.astype(np.float32)
