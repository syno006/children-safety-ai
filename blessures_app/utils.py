# blessures/utils.py
import base64
import io
import tempfile
import cv2
import numpy as np
from PIL import Image
from django.core.files.uploadedfile import UploadedFile

def process_uploaded_image(uploaded_file: UploadedFile):
    """
    Traite une image uploadée et retourne:
    - le chemin temporaire
    - l'image en array numpy
    - l'image en base64 pour l'affichage
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    
    # Lecture avec OpenCV
    img = cv2.imread(tmp_path)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(tmp_path).convert('RGB')), cv2.COLOR_RGB2BGR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Génération du base64 pour l'affichage
    _, buffer = cv2.imencode('.jpg', img)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'path': tmp_path,
        'array': img_rgb,
        'b64': image_b64
    }


def image_to_base64(img_array):
    """Convertit un array numpy en base64"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def draw_detection_boxes(img_array, boxes):
    """Dessine les bounding boxes sur l'image"""
    img_copy = img_array.copy()
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = box.get('confidence', 0)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f'{conf:.0%}', (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_copy


def apply_segmentation_overlay(img_array, mask):
    """Applique l'overlay de segmentation"""
    if mask is None or mask.size == 0:
        return img_array
    
    overlay = img_array.copy()
    overlay[mask > 0] = [255, 80, 80]  # Rouge
    return cv2.addWeighted(img_array, 0.55, overlay, 0.45, 0)