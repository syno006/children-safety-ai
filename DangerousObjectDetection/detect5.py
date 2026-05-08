"""
-----
  # Test on an image
  python detect_final.py --source photo.jpg

  # Test on a video
  python detect_final.py --source video.mp4

  # Live webcam
  python detect_final.py --source webcam

  # Custom weights path
  python detect_final.py --source photo.jpg --weights path/to/best.pt
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "best.pt"
DEFAULT_CONF    = 0.45
DEFAULT_IOU     = 0.45
IMG_SIZE        = 640

# ── Colors (BGR) ────────────────────────────────────────────────────────────
COLOR_BOX  = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── Load model ───────────────────────────────────────────────────────────────
def load_model(weights_path):
    print("\n  Loading YOLO model...")
    yolo = YOLO(weights_path)
    print("  Model ready!\n")
    return yolo


# ── Draw one detection ───────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BOX, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), FONT, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)


# ── Process a single frame ───────────────────────────────────────────────────
def process_frame(frame, yolo_model):
    results = yolo_model.predict(frame, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                                 imgsz=IMG_SIZE, verbose=False)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = yolo_model.names[cls]
            draw_box(frame, x1, y1, x2, y2, label, conf)

    return frame


# ── Image detection ──────────────────────────────────────────────────────────
def detect_image(yolo_model, source):
    print(f"  Analyzing image: {source}")
    frame = cv2.imread(source)
    if frame is None:
        print(f"  Cannot read image: {source}")
        return

    frame = process_frame(frame, yolo_model)
    print("  Done.")

    out_path = "result_" + Path(source).name
    cv2.imwrite(out_path, frame)
    print(f"   Saved → {out_path}")

    cv2.imshow("Weapon Detector", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Shared capture loop (webcam + video) ─────────────────────────────────────
def detect_capture(yolo_model, cap, output_path=None, is_video=False):
    writer     = None
    frame_time = time.time()

    # Set up video writer if saving output
    if is_video and output_path:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Writing output → {output_path}  ({total} frames)\n")

    print("  Running — press Q to quit\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, yolo_model)
        frame_idx += 1

        # FPS overlay
        now        = time.time()
        fps        = 1 / max(now - frame_time, 1e-6)
        frame_time = now
        h_f, w_f   = frame.shape[:2]
        cv2.putText(frame, f"FPS {fps:.1f}", (w_f - 110, h_f - 12),
                    FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow("Weapon Detector  (Q = quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Done.")


# ── Video detection ───────────────────────────────────────────────────────────
def detect_video(yolo_model, source):
    print(f"  Analyzing video: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  Cannot open video: {source}")
        return

    out_path = "result_" + Path(source).name
    detect_capture(yolo_model, cap, output_path=out_path, is_video=True)


# ── Webcam detection ──────────────────────────────────────────────────────────
def detect_webcam(yolo_model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open webcam")
        return
    detect_capture(yolo_model, cap, is_video=False)


# ── Main ──────────────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def main():
    parser = argparse.ArgumentParser(description="Weapon Detector")
    parser.add_argument("--source",  default="webcam",        help="image/video path | webcam")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to best.pt")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"  Weights not found: {args.weights}")
        print("   Make sure best.pt is in the same folder as this script")
        print("   or pass the correct path:  --weights path/to/best.pt")
        return

    yolo_model = load_model(args.weights)

    src = args.source.lower()
    ext = Path(src).suffix.lower()

    if src == "webcam":
        detect_webcam(yolo_model)
    elif ext in VIDEO_EXTS:
        detect_video(yolo_model, args.source)
    elif ext in IMAGE_EXTS:
        detect_image(yolo_model, args.source)
    else:
        print(f"  Unrecognized source: {args.source}")
        print(f"  Supported: webcam | image ({', '.join(IMAGE_EXTS)}) | video ({', '.join(VIDEO_EXTS)})")


if __name__ == "__main__":
    main()
