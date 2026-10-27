"""Readers for the public depth-evaluation sample format."""

import io
import json
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

PathLike = Union[str, os.PathLike]


def read_image(path: PathLike) -> np.ndarray:
    data = Path(path).read_bytes()
    encoded = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(path: PathLike) -> np.ndarray:
    """Read GeoUG's logarithmically encoded 16-bit depth PNG."""
    image = Image.open(io.BytesIO(Path(path).read_bytes()))
    if "near" not in image.info or "far" not in image.info:
        raise ValueError(
            f"Depth PNG must contain 'near' and 'far' metadata: {path}"
        )
    near = float(image.info["near"])
    far = float(image.info["far"])
    encoded = np.asarray(image)
    invalid = encoded == 0
    infinite = encoded == 65535
    normalized = (encoded.astype(np.float32) - 1.0) / 65533.0
    depth = near ** (1.0 - normalized) * far**normalized
    if "unit" in image.info:
        depth *= float(image.info["unit"])
    depth[invalid] = np.nan
    depth[infinite] = np.inf
    return depth


def read_json(path: PathLike):
    return json.loads(Path(path).read_text(encoding="utf-8"))
