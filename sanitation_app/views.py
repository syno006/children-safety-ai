"""
sanitation_app/views.py
"""

from django.shortcuts import render
from .model_loader import model_loader
from .utils import process_uploaded_image, numpy_to_b64


def home(request):
    """Landing page – overview of the sanitation risk module."""
    return render(request, "sanitation_app/home.html")


def detection(request):
    result      = None
    image_data  = None
    annotated   = None

    if request.method == "POST" and request.FILES.get("image"):
        img_info = process_uploaded_image(request.FILES["image"])
        
        # ── DEBUG ──────────────────────────────────────────
        from ultralytics import YOLO
        from pathlib import Path
        model = YOLO(str(Path(__file__).parent.parent / "models" / "sanitation" / "best.pt"))
        raw = model(img_info["array"], conf=0.01, verbose=True)[0]  # conf=0.01 = see everything
        print("=== RAW DETECTIONS ===")
        print("Model classes:", model.names)
        for box in raw.boxes:
            print(f"  cls={int(box.cls)} conf={float(box.conf):.3f} xyxy={box.xyxy[0].tolist()}")
        print("=== END ===")
        # ── END DEBUG ──────────────────────────────────────
        
        result   = model_loader.predict(img_info["array"])
        image_data = img_info["b64"]
        if result["annotated_image"] is not None:
            annotated = numpy_to_b64(result["annotated_image"])
        else:
            annotated = image_data

    return render(request, "sanitation_app/detection.html", {
        "result": result, "image_data": image_data, "annotated": annotated,
    })