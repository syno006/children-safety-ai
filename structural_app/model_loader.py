"""
structural_app/model_loader.py
Loads the YOLO11s structural damage model and runs inference.
Place your model file at:  models/structural/best.pt
"""

import os
import numpy as np
import cv2
from pathlib import Path

# ── Model path ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "structural" / "best.pt"

# ── Class config ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["Efflorescence", "spalling"]

# Risk weight per class (used for structural risk score)
CLASS_WEIGHTS = {
    "Efflorescence": 2.0,   # moisture / salt damage — moderate risk
    "spalling":      3.5,   # concrete breaking off — serious structural risk
}

# Colours for bounding boxes (BGR)
CLASS_COLORS = {
    "Efflorescence": (50,  160, 220),   # amber-blue
    "spalling":      (30,   60, 200),   # red
}

CONF_THRESHOLD = 0.25


class StructuralModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._load()
        return cls._instance

    # ── Loader ────────────────────────────────────────────────────────────────
    def _load(self):
        try:
            from ultralytics import YOLO
            if MODEL_PATH.exists():
                self._model = YOLO(str(MODEL_PATH))
                print(f"[StructuralLoader] Model loaded from {MODEL_PATH}")
            else:
                print(f"[StructuralLoader] WARNING – model not found at {MODEL_PATH}")
        except ImportError:
            print("[StructuralLoader] ultralytics not installed.")

    # ── Main inference ────────────────────────────────────────────────────────
    def predict(self, image_array: np.ndarray) -> dict:
        """
        Run YOLO11s on image_array (H×W×3 RGB uint8).
        Returns a rich result dict ready to be passed to the template.
        """
        if self._model is None:
            return self._empty_result()

        h, w = image_array.shape[:2]
        results = self._model(image_array, conf=CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf   = float(box.conf[0]) * 100          # → percentage
            cls_id = int(box.cls[0])
            label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            bw, bh = x2 - x1, y2 - y1
            area_pct = round((bw * bh) / (w * h) * 100, 2)

            detections.append({
                "label":           label,
                "confidence":      round(conf, 1),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width":  bw,
                "height": bh,
                "area_percentage": area_pct,
            })

        # ── Annotated image ───────────────────────────────────────────────────
        annotated = self._draw_boxes(image_array.copy(), detections)

        # ── Structural risk score (0–100, higher = more damage) ───────────────
        risk_score = self._compute_risk_score(detections, w * h)

        # ── Per-class counts ──────────────────────────────────────────────────
        class_counts = {c: 0 for c in CLASS_NAMES}
        for d in detections:
            if d["label"] in class_counts:
                class_counts[d["label"]] += 1

        return {
            "has_detections":  len(detections) > 0,
            "count":           len(detections),
            "detections":      detections,
            "class_counts":    class_counts,
            "risk_score":      risk_score,
            "risk_level":      self._risk_level(risk_score),
            "annotated_image": annotated,
            "total_area":      round(sum(d["area_percentage"] for d in detections), 1),
            "max_confidence":  round(max((d["confidence"] for d in detections), default=0), 1),
            "avg_confidence":  round(
                sum(d["confidence"] for d in detections) / len(detections)
                if detections else 0, 1
            ),
        }

    # ── Draw boxes ────────────────────────────────────────────────────────────
    def _draw_boxes(self, img_rgb: np.ndarray, detections: list) -> np.ndarray:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for d in detections:
            color = CLASS_COLORS.get(d["label"], (0, 200, 0))
            cv2.rectangle(img_bgr, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 3)
            label_txt = f"{d['label']}  {d['confidence']:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr,
                          (d["x1"], d["y1"] - th - 8),
                          (d["x1"] + tw + 6, d["y1"]),
                          color, -1)
            cv2.putText(img_bgr, label_txt,
                        (d["x1"] + 3, d["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Risk score ────────────────────────────────────────────────────────────
    def _compute_risk_score(self, detections: list, total_pixels: int) -> int:
        if not detections:
            return 0
        score = 0.0
        for d in detections:
            weight       = CLASS_WEIGHTS.get(d["label"], 1.0)
            conf_factor  = d["confidence"] / 100
            area_factor  = min(d["area_percentage"] / 20, 1.0)  # caps at 20% area
            score += weight * conf_factor * (1 + area_factor) * 20
        return min(int(score), 100)

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 50:
            return "critical"
        if score >= 25:
            return "moderate"
        return "low"

    @staticmethod
    def _empty_result() -> dict:
        return {
            "has_detections":  False,
            "count":           0,
            "detections":      [],
            "class_counts":    {"Efflorescence": 0, "spalling": 0},
            "risk_score":      0,
            "risk_level":      "low",
            "annotated_image": None,
            "total_area":      0,
            "max_confidence":  0,
            "avg_confidence":  0,
        }


# Singleton
model_loader = StructuralModelLoader()
