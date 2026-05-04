# blessures/views.py
import base64
import tempfile
import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from .model_loader import model_loader
from .utils import process_uploaded_image
from django.contrib import messages

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
        # Utiliser la méthode combinée pour une meilleure localisation
        result = model_loader.predict_with_combined_heatmap(img_info['array'])
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
        result = model_loader.predict_with_gradcam(img_info['array'])
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
            'detection': model_loader.predict_detection(img_info['array']),
            'segmentation': model_loader.predict_segmentation(img_info['array'])
        }
        
        image_data = img_info['b64']
        
        # Image avec détection
        if result['detection']['has_detections']:
            img_detection = img_info['array'].copy()
            for box in result['detection']['boxes']:
                cv2.rectangle(img_detection, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
                cv2.putText(img_detection, f"{box['confidence']:.0%}", (box['x1'], box['y1']-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_detection, cv2.COLOR_RGB2BGR))
            detection_image = base64.b64encode(buffer).decode('utf-8')
        else:
            detection_image = image_data
        
        # Image avec segmentation
        if result['segmentation']['has_segmentation']:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result['segmentation']['blended'], cv2.COLOR_RGB2BGR))
            segmentation_image = base64.b64encode(buffer).decode('utf-8')
        else:
            segmentation_image = image_data
    
    return render(request, 'blessures/full_pipeline.html', {
        'result': result,
        'image_data': image_data,
        'detection_image': detection_image,
        'segmentation_image': segmentation_image,
        'urgency_classes': URGENCY_CLASSES,
        'urgency_colors': URGENCY_COLORS
    })