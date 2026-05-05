import base64
import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from .model_loader import get_model
import logging

logger = logging.getLogger(__name__)

def home(request):
    """Page d'accueil ShelterCare"""
    return render(request, 'fall_detection/home.html')

def yolo_detection(request):
    """Détection YOLO avec bounding box"""
    
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Charger l'image
            image_file = request.FILES['image']
            img = Image.open(image_file)
            
            # Convertir pour OpenCV
            img_array = np.array(img.convert('RGB'))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Prédiction
            model = get_model()
            prediction_result = model.predict(img_array)
            
            if prediction_result.get('error'):
                return render(request, 'fall_detection/yolo.html', {
                    'error': prediction_result['error']
                })
            
            # Créer image avec bounding box
            img_display = img_array.copy()
            h, w = img_display.shape[:2]
            
            # Dessiner bounding box
            if prediction_result.get('box'):
                x1, y1, x2, y2 = prediction_result['box']
                
                # Ajuster les coordonnées à la taille originale
                x1 = int(x1 * w / 224)
                y1 = int(y1 * h / 224)
                x2 = int(x2 * w / 224)
                y2 = int(y2 * h / 224)
                
                # Choisir la couleur selon la prédiction
                if prediction_result['prediction'] == 'FALL':
                    color = (0, 0, 255)  # Rouge
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
                else:
                    color = (0, 255, 0)  # Vert
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                # Ajouter label
                label = f"{prediction_result['prediction']} ({prediction_result['confidence']:.1f}%)"
                cv2.putText(img_display, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Encoder l'image
            _, buffer = cv2.imencode('.jpg', img_display)
            image_data = base64.b64encode(buffer).decode()
            
            # Ajouter des conseils de sécurité
            safety_tips = []
            if prediction_result['prediction'] == 'FALL':
                safety_tips = [
                    "Alerter immédiatement le personnel du shelter",
                    "Ne pas déplacer l'enfant sans avis médical",
                    "Vérifier la conscience et la respiration"
                ]
            else:
                safety_tips = [
                    "Situation sécuritaire",
                    "Continuer la surveillance"
                ]
            
            result = {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'box': prediction_result.get('box'),
                'safety_tips': safety_tips,
            }
            
            context = {
                'result': result,
                'image_data': image_data,
            }
            
            return render(request, 'fall_detection/yolo.html', context)
            
        except Exception as e:
            logger.error(f"Erreur: {str(e)}")
            return render(request, 'fall_detection/yolo.html', {'error': str(e)})
    
    return render(request, 'fall_detection/yolo.html')

def classification_view(request):
    """Classification CNN sans bounding box"""
    
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            img = Image.open(image_file)
            img_array = np.array(img.convert('RGB'))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            model = get_model()
            prediction_result = model.predict(img_array)
            
            if prediction_result.get('error'):
                return render(request, 'fall_detection/classification.html', {
                    'error': prediction_result['error']
                })
            
            # Text overlay
            img_display = img_array.copy()
            if prediction_result['prediction'] == 'FALL':
                label = f"CHUTE DETECTEE - {prediction_result['confidence']:.1f}%"
                color = (0, 0, 255)
            else:
                label = f"SITUATION NORMALE - {prediction_result['confidence']:.1f}%"
                color = (0, 255, 0)
            
            cv2.putText(img_display, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            _, buffer = cv2.imencode('.jpg', img_display)
            image_data = base64.b64encode(buffer).decode()
            
            result = {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
            }
            
            return render(request, 'fall_detection/classification.html', {
                'result': result,
                'image_data': image_data
            })
            
        except Exception as e:
            logger.error(f"Erreur: {str(e)}")
            return render(request, 'fall_detection/classification.html', {'error': str(e)})
    
    return render(request, 'fall_detection/classification.html')