"""
structural_app/utils.py
Image helpers shared across views.
"""

import io
import base64
import numpy as np
from PIL import Image


def process_uploaded_image(django_file) -> dict:
    """
    Accepts a Django InMemoryUploadedFile / TemporaryUploadedFile.
    Returns:
        {
            "array": np.ndarray  (H×W×3 RGB uint8),
            "b64":   str         (base64-encoded JPEG for template display),
        }
    """
    raw   = django_file.read()
    img   = Image.open(io.BytesIO(raw)).convert("RGB")
    arr   = np.array(img)
    b64   = base64.b64encode(raw).decode("utf-8")
    return {"array": arr, "b64": b64}


def numpy_to_b64(img_rgb: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64-encoded JPEG string."""
    pil  = Image.fromarray(img_rgb.astype(np.uint8))
    buf  = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
