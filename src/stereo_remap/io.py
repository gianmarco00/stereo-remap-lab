"""I/O helpers for image reading and writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ArrayF = np.ndarray


def ensure_dir(path: Path) -> None:
    """Create a directory path when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def to_float_image(image: np.ndarray) -> ArrayF:
    """Convert image-like input to float32 in [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def read_image(path: str | Path) -> ArrayF:
    """Read an image file as RGB float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image from float [0, 1] or uint8 arrays."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(out_path)
