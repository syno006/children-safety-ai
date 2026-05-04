"""
sanitation_app/utils.py
Shared image helpers – mirrors blessures/utils.py style.
"""

import base64
import numpy as np
from PIL import Image
import io


def process_uploaded_image(uploaded_file) -> dict:
    """
    Takes a Django InMemoryUploadedFile, returns:
      { 'array': np.ndarray (RGB uint8),
        'b64':   str (base64 JPEG for template) }
    """
    pil_img = Image.open(uploaded_file).convert("RGB")

    # Cap large images to avoid slow inference
    max_size = 1024
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)

    img_array = np.array(pil_img, dtype=np.uint8)

    # Base64 for <img src="data:...">
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"array": img_array, "b64": b64}


def numpy_to_b64(img_rgb: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64 JPEG string."""
    import cv2
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")
