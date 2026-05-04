"""
Echo Heaven — Violence Detection Inference
Integrates EfficientNet-B4 + Bi-LSTM model into Django.
"""

import os
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 224
MAX_FRAMES = 16
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABEL_MAP  = {0: 'NonViolent', 1: 'Violent'}
THRESHOLD  = 0.50   # probability threshold for "Violent"

# ── Model architecture (must match training code exactly) ─────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):        # (B, T, H)
        w = torch.softmax(self.score(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1)


class EfficientLSTM(nn.Module):
    def __init__(self, hidden=256, num_classes=2, dropout=0.4):
        super().__init__()
        eff = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        for p in eff.features.parameters():
            p.requires_grad = False
        self.cnn  = eff.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1792, hidden, num_layers=2, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.attn = TemporalAttention(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):               # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        feats = self.pool(self.cnn(x.view(B * T, C, H, W))).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        ctx    = self.attn(out)
        return self.fc(self.drop(ctx))


# ── Singleton loader ──────────────────────────────────────────────────────────
_model = None

def load_model(weights_path: str) -> EfficientLSTM:
    """Load and cache the model. Call once at startup."""
    global _model
    if _model is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found at: {weights_path}. "
                "Set a valid MODEL_WEIGHTS_PATH or place the weights file there."
            )
        model = EfficientLSTM().to(DEVICE)
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _model = model
        print(f"[Echo Heaven] Model loaded from {weights_path} on {DEVICE}")
    return _model


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Extract evenly-spaced, normalised frames → float32 (T, H, W, 3)."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    frames, seen = [], -1
    for idx in indices:
        if idx != seen + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        seen = idx
        if not ret:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD
        frames.append(frame)
    cap.release()
    return np.stack(frames)          # (T, H, W, 3)


def extract_thumbnail_frame(video_path: str) -> np.ndarray | None:
    """Extract middle frame as uint8 RGB for display."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 360))
    return frame


def _heuristic_predict(video_path: str) -> dict:
    """
    Fallback prediction when trained weights are unavailable.
    Uses motion intensity between consecutive frames as a rough proxy signal.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)

    gray_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (320, 180))
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(gray_frames) < 2:
        # Conservative default when signal is too weak.
        prob_violent = 0.35
    else:
        diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
            diffs.append(float(np.mean(diff)) / 255.0)
        motion_score = float(np.mean(diffs))
        # Map motion score into a bounded probability range.
        prob_violent = float(np.clip(0.08 + (motion_score * 3.2), 0.05, 0.95))

    prob_safe = 1.0 - prob_violent
    label_int = int(prob_violent >= THRESHOLD)
    label = LABEL_MAP[label_int]

    if prob_violent >= 0.75:
        risk_level = "HIGH"
    elif prob_violent >= THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "label":        label,
        "label_int":    label_int,
        "prob_violent": round(prob_violent * 100, 2),
        "prob_safe":    round(prob_safe * 100, 2),
        "confidence":   round(max(prob_violent, prob_safe) * 100, 2),
        "risk_level":   risk_level,
        "alert":        label_int == 1,
    }


# ── Main inference function ───────────────────────────────────────────────────

def predict_video(video_path: str, weights_path: str) -> dict:
    """
    Run inference on a video file.

    Returns:
        {
            "label":      "Violent" | "NonViolent",
            "label_int":  1 | 0,
            "prob_violent": float,   # 0..1
            "prob_safe":    float,
            "confidence":   float,   # max(prob_violent, prob_safe)
            "risk_level":  "HIGH" | "MEDIUM" | "LOW",
            "alert":        bool,
        }
    """
    if not os.path.exists(weights_path):
        print(f"[Echo Heaven] Weights missing ({weights_path}). Running heuristic fallback.")
        return _heuristic_predict(video_path)

    model  = load_model(weights_path)
    frames = extract_frames(video_path)                      # (T, H, W, 3)
    tensor = torch.tensor(frames).permute(0, 3, 1, 2)       # (T, C, H, W)
    tensor = tensor.unsqueeze(0).float().to(DEVICE)          # (1, T, C, H, W)

    with torch.no_grad():
        logits = model(tensor)                               # (1, 2)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_safe    = float(probs[0])
    prob_violent = float(probs[1])
    label_int    = int(prob_violent >= THRESHOLD)
    label        = LABEL_MAP[label_int]

    if prob_violent >= 0.75:
        risk_level = "HIGH"
    elif prob_violent >= THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "label":        label,
        "label_int":    label_int,
        "prob_violent": round(prob_violent * 100, 2),
        "prob_safe":    round(prob_safe    * 100, 2),
        "confidence":   round(max(prob_violent, prob_safe) * 100, 2),
        "risk_level":   risk_level,
        "alert":        label_int == 1,
    }
# Add these three functions to violence_app/inference.py

def predict_main_model(video_path: str, weights_path: str) -> tuple[dict, float]:
    """Run main EfficientLSTM model, return (result_dict, prob_violent)"""
    result = predict_video(video_path, weights_path)
    return result, result['prob_violent'] / 100.0  # Convert % back to 0-1

def predict_yolo(video_path: str, weights_path: str) -> tuple[dict, float]:
    """Run YOLO model on video"""
    # Load YOLO model and run inference
    # Return (result_dict, prob_violent as 0-1 float)
    pass

def predict_mediapipe(video_path: str) -> tuple[dict, float]:
    """Run MediaPipe pose detection on video"""
    # Analyze poses and return violence risk score
    pass

def predict_ensemble(video_path: str, main_weights: str, yolo_weights: str) -> tuple[float, str]:
    """
    Ensemble prediction combining main model + YOLO + MediaPipe
    
    Returns:
        (risk_score, label): e.g., (0.65, 'Violent')
    """
    _, p_main = predict_main_model(video_path, main_weights)
    _, p_yolo = predict_yolo(video_path, yolo_weights)
    _, p_mp   = predict_mediapipe(video_path)
    
    risk = (p_main + p_yolo + p_mp) / 3
    label = 'Violent' if risk >= 0.5 else 'NonViolent'
    return risk, label