"""Model registry for the unified DangerousObjectDetection workspace."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from django.conf import settings

from .utils import load_image_from_path, numpy_to_b64


@dataclass
class ModelPipeline:
    id: str
    name: str
    description: str
    source_types: List[str]
    default_threshold: float
    run: Callable[..., Dict]


def _score_to_risk(score: float, threshold: float) -> Dict[str, str | bool]:
    if score >= threshold:
        return {"risk_level": "HIGH", "alert": True}
    if score >= threshold * 0.6:
        return {"risk_level": "MEDIUM", "alert": False}
    return {"risk_level": "LOW", "alert": False}


def _unavailable_result(reason: str) -> Dict:
    return {
        "label": "Unavailable",
        "score": 0,
        "confidence": 0,
        "risk_level": "LOW",
        "alert": False,
        "unavailable": True,
        "details": {"reason": reason},
    }


def run_pipeline(pipeline_id: str, source_type: str, *, media_path: Optional[str] = None, frame=None,
                 threshold: float = 50) -> Dict:
    pipeline = pipeline_map().get(pipeline_id)
    if pipeline is None:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")
    if source_type not in pipeline.source_types:
        return {
            "label": "Unsupported",
            "score": 0,
            "confidence": 0,
            "risk_level": "LOW",
            "alert": False,
            "details": {"reason": "unsupported-source"},
        }

    result = pipeline.run(source_type=source_type, media_path=media_path, frame=frame, threshold=threshold)
    score = float(result.get("score", 0))
    result.update(_score_to_risk(score, threshold))
    result["threshold"] = threshold
    return result


def pipeline_manifest() -> List[Dict]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "source_types": p.source_types,
            "default_threshold": p.default_threshold,
        }
        for p in pipelines()
    ]


def pipelines() -> List[ModelPipeline]:
    # Only include pipelines backed by local YOLO weights in the
    # models/FireDetec and models/dangerousOb folders as requested.
    return [
        _fire_detection_pipeline(),
        _dangerous_object_i_pipeline(),
    ]


def pipeline_map() -> Dict[str, ModelPipeline]:
    return {pipeline.id: pipeline for pipeline in pipelines()}


def _violence_video_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        from violence_app.inference import predict_video
        import cv2
        import numpy as np
        import tempfile
        import os

        weights_path = getattr(settings, 'MODEL_WEIGHTS_PATH', settings.VIOLENCE_MAIN_MODEL)
        if not os.path.exists(weights_path):
            return _unavailable_result("violence-weights-missing")

        if source_type == 'camera' and frame is not None:
            height, width = frame.shape[:2]
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_video_path = tmp.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            for _ in range(16):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            try:
                raw = predict_video(temp_video_path, weights_path)
            finally:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
        else:
            raw = predict_video(media_path, weights_path)

        return {
            "label": raw.get("label", "Unknown"),
            "score": float(raw.get("prob_violent", 0)),
            "confidence": float(raw.get("confidence", 0)),
            "details": raw,
        }

    return ModelPipeline(
        id="violence_video",
        name="Violence Detection (Video)",
        description="EfficientNet-B4 + Bi-LSTM ensemble for behavioral risk.",
        source_types=["video", "camera"],
        default_threshold=60,
        run=_run,
    )


def _sanitation_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        from sanitation_app.model_loader import model_loader

        if getattr(model_loader, "_model", None) is None:
            return _unavailable_result("sanitation-model-missing")

        image = frame if frame is not None else load_image_from_path(media_path)
        raw = model_loader.predict(image)

        annotated = raw.get('annotated_image')
        annotated_b64 = numpy_to_b64(annotated) if annotated is not None else None

        return {
            "label": raw.get("risk_level", "low").title(),
            "score": float(raw.get("risk_score", 0)),
            "confidence": float(raw.get("max_confidence", 0)),
            "details": raw,
            "artifact": annotated_b64,
        }

    return ModelPipeline(
        id="sanitation_risk",
        name="Sanitation Risk (Image)",
        description="YOLO11s sanitation detection (cockroach, trash, wet surface).",
        source_types=["image", "camera"],
        default_threshold=40,
        run=_run,
    )


def _injury_classification_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        from blessures_app.model_loader import model_loader

        if getattr(model_loader, "classifier", None) is None:
            return _unavailable_result("injury-classifier-missing")

        image = frame if frame is not None else load_image_from_path(media_path)
        raw = model_loader.predict_classification(image)

        prediction = raw.get('prediction', 'Unknown')
        confidence = float(raw.get('confidence_score', 0))
        score = confidence if prediction.lower().startswith('urgent') else max(0, 100 - confidence)

        return {
            "label": prediction,
            "score": score,
            "confidence": confidence,
            "details": raw,
        }

    return ModelPipeline(
        id="injury_classification",
        name="Injury Classification",
        description="Urgency classification with optional XAI outputs.",
        source_types=["image", "camera"],
        default_threshold=50,
        run=_run,
    )


def _injury_detection_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        from blessures_app.model_loader import model_loader
        from blessures_app.utils import draw_detection_boxes, image_to_base64

        if getattr(model_loader, "yolo_model", None) is None:
            return _unavailable_result("injury-detector-missing")

        image = frame if frame is not None else load_image_from_path(media_path)
        raw = model_loader.predict_detection_with_fallback(image)
        boxes = raw.get('boxes') or []
        artifact = image_to_base64(draw_detection_boxes(image, boxes)) if boxes else None
        return {
            "label": raw.get('method', 'detection').title(),
            "score": float(raw.get('max_confidence', 0)),
            "confidence": float(raw.get('avg_confidence', 0)),
            "details": raw,
            "artifact": artifact,
        }

    return ModelPipeline(
        id="injury_detection",
        name="Injury Detection",
        description="Bounding-box detection with YOLO + fallback pipeline.",
        source_types=["image", "camera"],
        default_threshold=40,
        run=_run,
    )


def _injury_segmentation_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        from blessures_app.model_loader import model_loader
        from blessures_app.utils import apply_segmentation_overlay, image_to_base64

        if getattr(model_loader, "seg_model", None) is None:
            return _unavailable_result("injury-segmentation-missing")

        image = frame if frame is not None else load_image_from_path(media_path)
        segmentations = model_loader.predict_segmentation_on_roi(image)

        if not segmentations:
            raw = model_loader.predict_segmentation(image)
            area_pct = float(raw.get('area_percentage', 0))
            details = raw
        else:
            seg = segmentations[0]
            area_pct = float(seg.get('area_percentage', 0))
            details = {
                'area_percentage': area_pct,
                'area_pixels': seg.get('area_pixels', 0),
                'roi_coverage': seg.get('roi_area_percentage', 0),
                'num_lesions': len(segmentations),
            }

        artifact = None
        if segmentations:
            blended = segmentations[0].get('blended')
            if blended is not None:
                artifact = image_to_base64(blended)
        elif details and isinstance(details, dict):
            mask = details.get('mask')
            if mask is not None:
                artifact = image_to_base64(apply_segmentation_overlay(image, mask))

        return {
            "label": "Segmentation",
            "score": area_pct,
            "confidence": area_pct,
            "details": details,
            "artifact": artifact,
        }

    return ModelPipeline(
        id="injury_segmentation",
        name="Injury Segmentation",
        description="Segmentation pipeline with ROI refinement.",
        source_types=["image", "camera"],
        default_threshold=8,
        run=_run,
    )


def _run_yolo_media(model_path: str, source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
    import os
    import numpy as np

    if not os.path.exists(model_path):
        return _unavailable_result("weights-missing")

    try:
        from ultralytics import YOLO
    except ImportError:
        return _unavailable_result("ultralytics-missing")

    model = YOLO(model_path)
    conf = max(0.01, min(0.99, float(threshold) / 100.0))

    try:
        import cv2
    except Exception:
        return _unavailable_result('opencv-missing')

    # Resolve the base image to run inference on.
    if frame is not None:
        # Camera / stream / video-worker already decoded the frame.
        image_rgb = frame
    elif source_type == 'video' and media_path:
        # Video file uploaded via run_source: extract the middle frame so we
        # get a representative sample rather than always frame 0 (often black).
        # Transcode AV1/HEVC/VP9 → H.264 using the same helper as views.py.
        # Import lazily to avoid circular imports (views imports registry).
        import subprocess, shutil as _shutil
        work_path = media_path
        try:
            _transcoded = media_path.rsplit('.', 1)[0] + '_h264.mp4'
            if os.path.exists(_transcoded):
                work_path = _transcoded  # Already transcoded by the worker.
            else:
                # Detect codec via ffprobe.
                _codec = ''
                if _shutil.which('ffprobe'):
                    _r = subprocess.run(
                        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=codec_name',
                         '-of', 'default=noprint_wrappers=1:nokey=1', media_path],
                        capture_output=True, text=True, timeout=15,
                    )
                    _codec = _r.stdout.strip().lower()
                _bad_codecs = {'av1', 'hevc', 'vp9', 'vp8', 'av01'}
                if _codec in _bad_codecs or (_codec == '' and _shutil.which('ffmpeg')):
                    if _shutil.which('ffmpeg'):
                        _r2 = subprocess.run(
                            ['ffmpeg', '-y', '-i', media_path, '-c:v', 'libx264',
                             '-preset', 'fast', '-crf', '23', '-c:a', 'aac',
                             '-movflags', '+faststart', _transcoded],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
                        )
                        if _r2.returncode == 0 and os.path.exists(_transcoded):
                            work_path = _transcoded
        except Exception:
            pass
        cap = cv2.VideoCapture(work_path)
        if not cap.isOpened():
            return _unavailable_result('unable-to-open-video')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            return _unavailable_result('unable-to-read-video-frame')
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        # Image file.
        image_rgb = load_image_from_path(media_path)

    if image_rgb is None:
        return _unavailable_result('unable-to-load-image')

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(frame_bgr, conf=conf, iou=0.45, imgsz=640, verbose=False)

    detections: list[dict] = []
    max_conf = 0.0
    names = getattr(model, 'names', None) or {}

    if results:
        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for box in r.boxes:
                # xyxy is shape (1,4)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c = float(box.conf[0])
                cls = int(box.cls[0])
                label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                max_conf = max(max_conf, c)

                detections.append({
                    'label': label,
                    'confidence': c,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': cls,
                })

                # Draw box + label like the script (BGR colors)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = f"{label} {c:.0%}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 255), -1)
                cv2.putText(frame_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    score = float(max_conf * 100.0)
    artifact = numpy_to_b64(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    return {
        "label": "Detected" if score > 0 else "None",
        "score": score,
        "confidence": score,
        "details": {"detections": int(len(detections))},
        "detections": detections,
        "artifact": artifact,
    }


def annotate_frame_bgr_with_yolo(model_path: str, frame_bgr, *, threshold: float = 50) -> tuple:
    """Detect on a single BGR frame and draw boxes onto the SAME frame.

    This mirrors the user's detect_final.py behavior:
    - input: OpenCV BGR frame
    - model.predict(frame)
    - draw boxes in-place (BGR)

    Returns (frame_bgr, detections, max_conf)
    """

    import os

    if frame_bgr is None:
        return frame_bgr, [], 0.0
    if not os.path.exists(model_path):
        return frame_bgr, [], 0.0

    try:
        import cv2
    except Exception:
        return frame_bgr, [], 0.0

    # Cache models by path so we don't reload weights for every frame.
    try:
        from ultralytics import YOLO
    except Exception:
        return frame_bgr, [], 0.0

    global _YOLO_MODEL_CACHE  # type: ignore
    try:
        _YOLO_MODEL_CACHE
    except NameError:
        _YOLO_MODEL_CACHE = {}

    model = _YOLO_MODEL_CACHE.get(model_path)
    if model is None:
        model = YOLO(model_path)
        _YOLO_MODEL_CACHE[model_path] = model
    conf = max(0.01, min(0.99, float(threshold) / 100.0))

    results = model.predict(frame_bgr, conf=conf, iou=0.45, imgsz=640, verbose=False)
    names = getattr(model, 'names', None) or {}

    detections: list[dict] = []
    max_conf = 0.0

    if results:
        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                c = float(box.conf[0])
                cls = int(box.cls[0])
                label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                max_conf = max(max_conf, c)
                detections.append({
                    'label': label,
                    'confidence': c,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': cls,
                })

                # Draw like detect_final.py
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = f"{label} {c:.0%}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 255), -1)
                cv2.putText(frame_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr, detections, max_conf


def annotate_frame_bgr_with_two_yolos(
    model_a_path: str,
    model_b_path: str,
    frame_bgr,
    *,
    threshold_a: float = 50,
    threshold_b: float = 50,
) -> tuple:
    """Run and draw TWO YOLO models onto the same BGR frame (in-place).

    Returns (frame_bgr, detections_by_pipeline, max_conf_overall)
    where detections_by_pipeline is a dict with keys: 'a', 'b'.
    """

    dets_by: dict[str, list[dict]] = {}
    max_conf = 0.0

    frame_bgr, dets_a, max_a = annotate_frame_bgr_with_yolo(model_a_path, frame_bgr, threshold=threshold_a)
    if dets_a:
        dets_by['a'] = dets_a
    max_conf = max(max_conf, float(max_a or 0.0))

    frame_bgr, dets_b, max_b = annotate_frame_bgr_with_yolo(model_b_path, frame_bgr, threshold=threshold_b)
    if dets_b:
        dets_by['b'] = dets_b
    max_conf = max(max_conf, float(max_b or 0.0))

    return frame_bgr, dets_by, max_conf


def _fire_detection_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        model_path = str(settings.BASE_DIR / "models" / "FireDetec" / "bestF.pt")
        result = _run_yolo_media(model_path, source_type, media_path, frame, threshold)
        if result.get("unavailable"):
            result["details"]["reason"] = "fire-model-missing"
        return result

    return ModelPipeline(
        id="fire_detection",
        name="Fire Detection",
        description="YOLO model for fire/smoke detection.",
        source_types=["image", "video", "camera"],
        default_threshold=45,
        run=_run,
    )


def _dangerous_object_a_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        model_path = str(settings.BASE_DIR / "models" / "dangerousOb" / "bestA.pt")
        result = _run_yolo_media(model_path, source_type, media_path, frame, threshold)
        if result.get("unavailable"):
            result["details"]["reason"] = "dangerousob-bestA-missing"
        return result

    return ModelPipeline(
        id="dangerous_object_a",
        name="Dangerous Objects A",
        description="Custom YOLO model bestA.pt (dangerous objects).",
        source_types=["image", "video", "camera"],
        default_threshold=40,
        run=_run,
    )


def _dangerous_object_i_pipeline() -> ModelPipeline:
    def _run(source_type: str, media_path: Optional[str], frame=None, threshold: float = 50) -> Dict:
        model_path = str(settings.BASE_DIR / "models" / "dangerousOb" / "bestI.pt")
        result = _run_yolo_media(model_path, source_type, media_path, frame, threshold)
        if result.get("unavailable"):
            result["details"]["reason"] = "dangerousob-bestI-missing"
        return result

    return ModelPipeline(
        id="dangerous_object_i",
        name="Dangerous Objects I",
        description="Custom YOLO model bestI.pt (dangerous objects).",
        source_types=["image", "video", "camera"],
        default_threshold=40,
        run=_run,
    )
