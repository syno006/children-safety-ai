"""Shared utilities for DangerousObjectDetection."""
from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_from_path(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert('RGB'))


def load_image_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.array(img.convert('RGB'))


def decode_base64_image(data_url: str) -> np.ndarray:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    raw = base64.b64decode(data_url)
    return load_image_from_bytes(raw)


def numpy_to_b64(img_rgb: np.ndarray) -> str:
    import cv2

    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')
