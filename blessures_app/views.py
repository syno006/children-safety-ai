# blessures/views.py
import base64
import os
import tempfile
import uuid
import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from .model_loader import model_loader
from .utils import process_uploaded_image
from django.contrib import messages  # Ajouter cet import

# Constantes
URGENCY_CLASSES = ['Urgent', 'Non-Urgent', 'Incertain']
URGENCY_COLORS = {0: '#e74c3c', 1: '#27ae60', 2: '#f39c12'}
URGENT_THRESHOLD = 0.35
YOLO_CONF_THRESHOLD = 0.20


def home(request):
    """Page d'accueil"""
    return render(request, 'blessures/home.html', {
        'urgency_classes': URGENCY_CLASSES,
        'urgency_colors': URGENCY_COLORS,
    })


def classification_view(request):
    """Classification avec heatmap combinée (Grad-CAM + Détection + Segmentation)"""
    result = None
    image_data = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        img_info = process_uploaded_image(request.FILES['image'])
        result = model_loader.predict_classification(img_info['array'])
        image_data = img_info['b64']
    
    return render(request, 'blessures/classification.html', {
        'result': result,
        'image_data': image_data,
        'threshold': URGENT_THRESHOLD
    })
def classification_xai_view(request):
    """Classification avec explications XAI (LRP, Grad-CAM)"""
    result = None
    image_data = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        img_info = process_uploaded_image(request.FILES['image'])
        result = model_loader.predict_with_xai(img_info['array'])
        image_data = img_info['b64']
    
    return render(request, 'blessures/classification_xai.html', {
        'result': result,
        'image_data': image_data,
        'threshold': URGENT_THRESHOLD
    })


def detection_view(request):
    """Détection des lésions avec bounding boxes visibles"""
    result = None
    image_data = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        from django.contrib import messages
        import base64
        import cv2
        
        img_info = process_uploaded_image(request.FILES['image'])
        
        # Utiliser la méthode avec fallback
        result = model_loader.predict_detection_with_fallback(img_info['array'])
        
        # Dessiner les bounding boxes sur l'image
        img_with_boxes = img_info['array'].copy()
        
        # Choisir la couleur selon la méthode utilisée
        if result.get('method') == 'yolo':
            color = (0, 255, 0)  # Vert pour YOLO
        elif result.get('method') == 'segmentation':
            color = (255, 165, 0)  # Orange pour segmentation
            messages.info(request, "🔍 Détection via segmentation")
        else:
            color = (0, 100, 255)  # Bleu-orange pour défaut
            messages.warning(request, "⚠️ Détection par défaut")
        
        for box in result['boxes']:
            # Dessiner le rectangle
            cv2.rectangle(img_with_boxes, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 3)
            
            # Ajouter le texte de confiance
            label = f"{box['confidence']:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_with_boxes, (box['x1'], box['y1'] - label_size[1] - 5), 
                         (box['x1'] + label_size[0] + 5, box['y1']), color, -1)
            cv2.putText(img_with_boxes, label, (box['x1'] + 2, box['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convertir en base64 avec les boxes
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        image_data = base64.b64encode(buffer).decode('utf-8')
    
    return render(request, 'blessures/detection.html', {
        'result': result,
        'image_data': image_data,
        'conf_threshold': YOLO_CONF_THRESHOLD
    })

def segmentation_view(request):
    """Segmentation des plaies avec fallback"""
    result = None
    image_data = None
    mask_data = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        from django.contrib import messages
        
        img_info = process_uploaded_image(request.FILES['image'])
        
        # Utiliser la segmentation ROI qui a un fallback
        segmentations = model_loader.predict_segmentation_on_roi(img_info['array'])
        
        if segmentations and len(segmentations) > 0:
            # Prendre la première segmentation
            seg = segmentations[0]
            result = {
                'has_segmentation': True,
                'area_percentage': seg['area_percentage'],
                'area_pixels': seg['area_pixels'],
                'num_lesions': len(segmentations),
                'roi_coverage': seg['roi_area_percentage']
            }
            # Convertir l'overlay en base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(seg['blended'], cv2.COLOR_RGB2BGR))
            mask_data = base64.b64encode(buffer).decode('utf-8')
            
            messages.success(request, f"✅ {len(segmentations)} blessure(s) segmentée(s)")
        else:
            # Fallback: utiliser la segmentation simple
            result = model_loader.predict_segmentation(img_info['array'])
            if not result['has_segmentation']:
                # Dernier recours: simulation
                result = model_loader._simulate_segmentation(img_info['array'])
                messages.warning(request, "⚠️ Utilisation de la visualisation par défaut")
            
            if result.get('blended') is not None:
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result['blended'], cv2.COLOR_RGB2BGR))
                mask_data = base64.b64encode(buffer).decode('utf-8')
        
        image_data = img_info['b64']
    
    return render(request, 'blessures/segmentation.html', {
        'result': result,
        'image_data': image_data,
        'mask_data': mask_data
    })
def segmentation_roi_view(request):
    """Segmentation UNIQUEMENT sur la région de la blessure"""
    result = None
    image_data = None
    segmentation_results = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        import base64
        import cv2
        
        img_info = process_uploaded_image(request.FILES['image'])
        
        # Segmentation ciblée sur les régions d'intérêt
        segmentation_results = model_loader.predict_segmentation_on_roi(img_info['array'])
        
        if segmentation_results:
            # Prendre la première segmentation pour l'affichage principal
            main_seg = segmentation_results[0]
            
            result = {
                'has_segmentation': True,
                'area_percentage': main_seg['area_percentage'],
                'area_pixels': main_seg['area_pixels'],
                'roi_area_percentage': main_seg['roi_area_percentage'],
                'num_lesions': len(segmentation_results)
            }
            
            # Image avec overlay (segmentation + boîtes)
            img_with_boxes_and_seg = img_info['array'].copy()
            
            for seg in segmentation_results:
                # Dessiner la boîte de détection
                box = seg['box']
                cv2.rectangle(img_with_boxes_and_seg, 
                            (box['x1'], box['y1']), 
                            (box['x2'], box['y2']), 
                            (0, 255, 0), 2)
                
                # Ajouter la segmentation en overlay
                mask = seg['mask']
                img_with_boxes_and_seg[mask > 0] = cv2.addWeighted(
                    img_with_boxes_and_seg[mask > 0], 0.6,
                    np.array([255, 80, 80]), 0.4, 0
                )
                
                # Ajouter les contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_with_boxes_and_seg, contours, -1, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_with_boxes_and_seg, cv2.COLOR_RGB2BGR))
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Images individuelles pour chaque blessure
            lesion_images = []
            for i, seg in enumerate(segmentation_results):
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(seg['blended'], cv2.COLOR_RGB2BGR))
                lesion_images.append({
                    'id': i + 1,
                    'image': base64.b64encode(buffer).decode('utf-8'),
                    'area': seg['area_percentage'],
                    'roi_area': seg['roi_area_percentage']
                })
        else:
            result = {'has_segmentation': False}
            image_data = img_info['b64']
            lesion_images = []
    
    return render(request, 'blessures/segmentation_roi.html', {
        'result': result,
        'image_data': image_data,
        'segmentation_results': segmentation_results,
        'lesion_images': lesion_images if 'lesion_images' in locals() else []
    })


# Dans full_pipeline_view, modifiez l'appel de détection :
def full_pipeline_view(request):
    """Pipeline complet"""
    result = None
    image_data = None
    detection_image = None
    segmentation_image = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        img_info = process_uploaded_image(request.FILES['image'])
        
        result = {
            'classification': model_loader.predict_classification(img_info['array']),
            'detection': model_loader.predict_detection_with_fallback(img_info['array']),  # ← Utiliser with_fallback
        }
        
        image_data = img_info['b64']
        
        # Segmentation sur ROI prioritaire, puis fallback
        segmentation_results = model_loader.predict_segmentation_on_roi(img_info['array'])
        if segmentation_results and len(segmentation_results) > 0:
            main_seg = segmentation_results[0]
            combined_overlay = img_info['array'].copy()
            for seg in segmentation_results:
                mask = seg['mask']
                combined_overlay[mask > 0] = [255, 80, 80]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined_overlay, contours, -1, (0, 255, 0), 2)
            combined_blended = cv2.addWeighted(img_info['array'], 0.55, combined_overlay, 0.45, 0)
            result['segmentation'] = {
                'has_segmentation': True,
                'area_percentage': main_seg['area_percentage'],
                'area_pixels': main_seg['area_pixels'],
                'num_lesions': len(segmentation_results),
                'roi_area_percentage': main_seg['roi_area_percentage'],
                'blended': combined_blended
            }
            segmentation_image = base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(combined_blended, cv2.COLOR_RGB2BGR))[1]).decode('utf-8')
        else:
            fallback_seg = model_loader.predict_segmentation(img_info['array'])
            if not fallback_seg['has_segmentation']:
                fallback_seg = model_loader._simulate_segmentation(img_info['array'])
            result['segmentation'] = fallback_seg
            if fallback_seg.get('blended') is not None:
                segmentation_image = base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(fallback_seg['blended'], cv2.COLOR_RGB2BGR))[1]).decode('utf-8')
            else:
                segmentation_image = image_data
        
        # Image avec détection - utiliser les boxes du résultat avec fallback
        if result['detection']['has_detections']:
            img_detection = img_info['array'].copy()
            for box in result['detection']['boxes']:
                cv2.rectangle(img_detection, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
                cv2.putText(img_detection, f"{box['confidence']:.0f}%", (box['x1'], box['y1']-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_detection, cv2.COLOR_RGB2BGR))
            detection_image = base64.b64encode(buffer).decode('utf-8')
        else:
            detection_image = image_data
    
    return render(request, 'blessures/full_pipeline.html', {
        'result': result,
        'image_data': image_data,
        'detection_image': detection_image,
        'segmentation_image': segmentation_image,
        'urgency_classes': URGENCY_CLASSES,
        'urgency_colors': URGENCY_COLORS
    })

def video_analysis_view(request):
    result = None
    output_video_url = None

    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Sauvegarder le fichier d'entrée
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
            for chunk in video_file.chunks():
                tmp_in.write(chunk)
            tmp_in_path = tmp_in.name

        # Fichier de sortie dans MEDIA_ROOT
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'video_results'), exist_ok=True)
        output_filename = f"result_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(settings.MEDIA_ROOT, 'video_results', output_filename)

        try:
            cap = cv2.VideoCapture(tmp_in_path)
            if not cap.isOpened():
                raise ValueError("Impossible d'ouvrir la vidéo")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0

            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"📹 Vidéo: {W}x{H} @ {fps:.1f}fps, {total} frames")

            # VideoWriter avec codec mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            if not out.isOpened():
                raise ValueError("Impossible de créer le VideoWriter")

            PROCESS_EVERY = 3
            MAX_FRAMES = min(300, total)
            frame_idx = 0
            last_annotated = None
            stats = {
                'urgent': 0, 'non_urgent': 0, 'incertain': 0,
                'frames_processed': 0,
                'total_detections': 0,
                'all_confidences': [],
                'max_confidence': 0.0,
            }

            while frame_idx < MAX_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_idx % PROCESS_EVERY == 0:
                    try:
                        annotated_rgb = _annotate_frame(frame_rgb, stats)
                        last_annotated = annotated_rgb
                        stats['frames_processed'] += 1
                        print(f"✅ Frame {frame_idx}")
                    except Exception as e:
                        print(f"❌ Frame {frame_idx}: {e}")
                        annotated_rgb = frame_rgb
                else:
                    annotated_rgb = last_annotated if last_annotated is not None else frame_rgb

                out.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                frame_idx += 1

            cap.release()
            out.release()
            print(f"🎬 Terminé: {frame_idx} frames, fichier: {os.path.getsize(output_path)} bytes")

            # ── Re-encoder avec ffmpeg pour compatibilité navigateur ──
            reencoded_path = output_path.replace('.mp4', '_web.mp4')
            ffmpeg_cmd = (
                f'ffmpeg -y -i "{output_path}" '
                f'-vcodec libx264 -pix_fmt yuv420p '
                f'-movflags +faststart '
                f'"{reencoded_path}" 2>/dev/null'
            )
            exit_code = os.system(ffmpeg_cmd)

            if exit_code == 0 and os.path.exists(reencoded_path):
                os.unlink(output_path)
                final_path = reencoded_path
                final_filename = output_filename.replace('.mp4', '_web.mp4')
                print("✅ Re-encodage ffmpeg réussi")
            else:
                # ffmpeg non disponible — garder le fichier mp4v
                final_path = output_path
                final_filename = output_filename
                print("⚠️ ffmpeg non disponible, fichier mp4v conservé")

            output_video_url = settings.MEDIA_URL + 'video_results/' + final_filename

            if stats['all_confidences']:
                avg_conf = sum(stats['all_confidences']) / len(stats['all_confidences'])
            else:
                avg_conf = 0.0

            result = {
                'total_frames': frame_idx,
                'frames_processed': stats['frames_processed'],
                'urgent_frames': stats['urgent'],
                'non_urgent_frames': stats['non_urgent'],
                'incertain_frames': stats['incertain'],
                'total_detections': stats['total_detections'],
                'avg_confidence': avg_conf,
                'max_confidence': stats['max_confidence'],
                'dominant_class': max(
                    ['Urgent', 'Non-Urgent', 'Incertain'],
                    key=lambda c: stats[c.lower().replace('-', '_')]
                ),
                'output_filename': final_filename,
            }

        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback; traceback.print_exc()
            result = {'error': str(e)}

        finally:
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)

    return render(request, 'blessures/video_analysis.html', {
        'result': result,
        'output_video_url': output_video_url,
        'urgency_colors': URGENCY_COLORS,
    })


def download_video(request, filename):
    """Téléchargement direct de la vidéo annotée"""
    from django.http import FileResponse, Http404
    filepath = os.path.join(settings.MEDIA_ROOT, 'video_results', filename)
    if not os.path.exists(filepath):
        raise Http404("Vidéo non trouvée")
    return FileResponse(
        open(filepath, 'rb'),
        content_type='video/mp4',
        as_attachment=True,
        filename=f"analyse_blessure_{filename}"
    )

def _annotate_frame(frame_rgb, stats):
    """
    Classification + Détection YOLO UNIQUEMENT — pas de segmentation.
    """
    annotated = frame_rgb.copy()
    h, w = frame_rgb.shape[:2]

    # ── 1. CLASSIFICATION ─────────────────────────────────
    clf        = model_loader.predict_classification(frame_rgb)
    label      = clf['prediction']
    confidence = clf['confidence_score']

    key = label.lower().replace('-', '_')
    if key in stats:
        stats[key] += 1

    color_map = {
        'Urgent':     (231, 76,  60),
        'Non-Urgent': (39,  174, 96),
        'Incertain':  (243, 156, 18),
    }
    color_clf = color_map.get(label, (200, 200, 200))

    # Bandeau supérieur
    banner = annotated.copy()
    cv2.rectangle(banner, (0, 0), (w, 52), (15, 15, 15), -1)
    cv2.addWeighted(banner, 0.78, annotated, 0.22, 0, annotated)
    cv2.putText(annotated,
                f"CLF : {label}   {confidence:.0f}%",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_clf, 2, cv2.LINE_AA)

    # ── 2. DETECTION YOLO (sans fallback) ─────────────────
    det = model_loader.predict_detection(frame_rgb)

    if det['has_detections']:
        stats['total_detections'] += len(det['boxes'])
        for box in det['boxes']:
            stats['all_confidences'].append(box['confidence'])
            stats['max_confidence'] = max(stats['max_confidence'], box['confidence'])

    for box in det['boxes']:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)

        label_txt = f"Blessure  {box['confidence']:.0f}%"
        (tw, th), _ = cv2.getTextSize(
            label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        ty = max(y1 - 8, 60)
        cv2.rectangle(annotated,
                      (x1, ty - th - 6), (x1 + tw + 8, ty + 2),
                      (255, 165, 0), -1)
        cv2.putText(annotated, label_txt,
                    (x1 + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)

    # ── 3. Bandeau bas ─────────────────────────────────────
    det_txt = (f"YOLO : {det['count']} blessure(s) detectee(s)"
               if det['has_detections'] else "YOLO : aucune detection")

    footer = annotated.copy()
    cv2.rectangle(footer, (0, h - 34), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(footer, 0.78, annotated, 0.22, 0, annotated)
    cv2.putText(annotated, det_txt,
                (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated