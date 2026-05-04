"""
Echo Heaven — Django Views
Handles video upload + violence prediction.
"""
import os
import json
import uuid
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .inference import predict_video
from .models import AnalysisRecord   # see models.py below


def index(request):
    """Landing / dashboard page."""
    recent = AnalysisRecord.objects.order_by('-created_at')[:10]
    stats  = _compute_stats()
    return render(request, 'detector/index.html', {
        'recent': recent,
        'stats':  stats,
    })


def upload_view(request):
    """Video upload page."""
    return render(request, 'detector/upload.html')


def analyze(request):
    """
    POST: accept a video file, run inference, save result, redirect to result page.
    GET:  redirect to upload page.
    """
    if request.method != 'POST':
        return redirect('upload')

    video_file = request.FILES.get('video')
    if not video_file:
        return render(request, 'detector/upload.html', {'error': 'Please select a video file.'})

    # ── Validate file type ────────────────────────────────────────────────────
    allowed_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = Path(video_file.name).suffix.lower()
    if ext not in allowed_exts:
        return render(request, 'detector/upload.html',
                      {'error': f'Unsupported format ({ext}). Use: {", ".join(allowed_exts)}'})

    # ── Save uploaded file ────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_dir  = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    fs       = FileSystemStorage(location=upload_dir)
    filename = fs.save(unique_name, video_file)
    video_path = os.path.join(upload_dir, filename)

    # ── Run inference ─────────────────────────────────────────────────────────
    weights_path = getattr(settings, 'MODEL_WEIGHTS_PATH', settings.VIOLENCE_MAIN_MODEL)
    try:
        result = predict_video(video_path, weights_path)
    except Exception as e:
        return render(request, 'detector/upload.html',
                      {'error': f'Analysis failed: {str(e)}'})

    # ── Persist to DB ─────────────────────────────────────────────────────────
    record = AnalysisRecord.objects.create(
        original_filename = video_file.name,
        stored_filename   = unique_name,
        label             = result['label'],
        prob_violent      = result['prob_violent'],
        prob_safe         = result['prob_safe'],
        confidence        = result['confidence'],
        risk_level        = result['risk_level'],
        alert             = result['alert'],
    )

    return redirect('result', pk=record.pk)


def result_view(request, pk):
    """Show analysis result."""
    record = get_object_or_404(AnalysisRecord, pk=pk)
    video_url = f"{settings.MEDIA_URL}uploads/{record.stored_filename}"
    return render(request, 'detector/result.html', {
        'record':    record,
        'video_url': video_url,
    })


def history_view(request):
    """All past analyses."""
    records = AnalysisRecord.objects.order_by('-created_at')
    return render(request, 'detector/history.html', {'records': records})


# ── AJAX endpoint ─────────────────────────────────────────────────────────────

def api_analyze(request):
    """JSON API endpoint for async upload."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No file'}, status=400)

    ext = Path(video_file.name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_dir  = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    fs = FileSystemStorage(location=upload_dir)
    filename   = fs.save(unique_name, video_file)
    video_path = os.path.join(upload_dir, filename)

    weights_path = getattr(settings, 'MODEL_WEIGHTS_PATH', settings.VIOLENCE_MAIN_MODEL)
    try:
        result = predict_video(video_path, weights_path)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    record = AnalysisRecord.objects.create(
        original_filename = video_file.name,
        stored_filename   = unique_name,
        **result
    )
    result['id']        = record.pk
    result['video_url'] = f"{settings.MEDIA_URL}uploads/{unique_name}"
    return JsonResponse(result)

@require_POST
def api_analyze_frame(request):
    """API endpoint for real-time frame analysis."""
    frame_file = request.FILES.get('frame')
    if not frame_file:
        return JsonResponse({'error': 'No frame provided'}, status=400)

    # Save the frame temporarily
    import tempfile
    import cv2
    import numpy as np
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        for chunk in frame_file.chunks():
            tmp.write(chunk)
        frame_path = tmp.name

    try:
        # Read the frame
        img = cv2.imread(frame_path)
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(frame_path).convert('RGB')), cv2.COLOR_RGB2BGR)

        # For real-time analysis, we'll use a simplified approach
        # Create a short "video" by duplicating the frame
        temp_video_path = frame_path.replace('.jpg', '.mp4')
        height, width = img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))

        # Write the same frame multiple times to simulate a short video
        for _ in range(16):  # Match the model's expected frame count
            out.write(img)
        out.release()

        # Run inference on the temporary video
        weights_path = getattr(settings, 'MODEL_WEIGHTS_PATH', settings.VIOLENCE_MAIN_MODEL)
        result = predict_video(temp_video_path, weights_path)

        # Clean up
        os.unlink(temp_video_path)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        if os.path.exists(frame_path):
            os.unlink(frame_path)

def detection_view(request):
    return render(request, 'detector/detection.html')
# ── Helpers ───────────────────────────────────────────────────────────────────

def get_metrics(request):
    path = os.path.join(settings.BASE_DIR, 'metrics', 'metrics.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return JsonResponse(data)
    except FileNotFoundError:
        return JsonResponse({'error': 'metrics.json introuvable. Lance extract_metrics.py d\'abord.'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'metrics.json corrompu.'}, status=500)
    
def _compute_stats():
    total   = AnalysisRecord.objects.count()
    violent = AnalysisRecord.objects.filter(label='Violent').count()
    safe    = total - violent
    high    = AnalysisRecord.objects.filter(risk_level='HIGH').count()
    return {
        'total':   total,
        'violent': violent,
        'safe':    safe,
        'high':    high,
        'pct_violent': round(violent / total * 100, 1) if total else 0,
    }
