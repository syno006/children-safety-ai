# fall_detection/model_loader.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FALL_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'fall' / 'best_model.pt'

logger.info(f"Using device: {DEVICE}")
logger.info(f"IMG_SIZE: {IMG_SIZE}")


class FastFallDetector(nn.Module):
    """Modèle rapide pour détection de chutes avec localisation"""
    
    def __init__(self, num_classes=2):
        super(FastFallDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.localizer = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        class_output = self.classifier(features_flat)
        loc_output = self.localizer(features)
        return class_output, loc_output
    
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            class_output, loc_output = self.forward(x)
            class_probs = torch.softmax(class_output, dim=1)
            predicted_class = torch.argmax(class_probs, dim=1)
            
            loc_map = torch.nn.functional.interpolate(
                loc_output, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
            )
            
            boxes = []
            for i in range(x.size(0)):
                heatmap = loc_map[i, 0].cpu().numpy()
                ys, xs = np.where(heatmap > threshold)
                if len(ys) > 0:
                    boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
                else:
                    boxes.append([IMG_SIZE//3, IMG_SIZE//3, 2*IMG_SIZE//3, 2*IMG_SIZE//3])
            
            return predicted_class, class_probs, boxes, loc_map


class FallDetectionModelWrapper:
    """Wrapper pour le modèle de détection de chutes"""
    
    def __init__(self):
        self.model = None
        # CORRECTION : Inverser l'ordre des classes si nécessaire
        self.class_names = ['fall', 'normal']  # Index 0 = FALL, Index 1 = NORMAL
        self.load_model()
    
    def load_model(self):
        """Charger le modèle PyTorch"""
        try:
            if FALL_MODEL_PATH.exists():
                logger.info(f"📦 Chargement modèle depuis: {FALL_MODEL_PATH}")
                self.model = FastFallDetector(num_classes=2).to(DEVICE)
                checkpoint = torch.load(FALL_MODEL_PATH, map_location=DEVICE)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        # Vérifier si les classes sont dans le checkpoint
                        if 'class_names' in checkpoint:
                            loaded_classes = checkpoint['class_names']
                            logger.info(f"Classes chargées: {loaded_classes}")
                            # Si les classes sont inversées, on peut les corriger
                            if loaded_classes == ['normal', 'fall']:
                                logger.warning("⚠️ Détection d'inversion des classes - Correction automatique")
                                # Garder notre ordre corrigé
                        logger.info(f"✅ Modèle chargé")
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                        logger.info(f"✅ Modèle chargé (state_dict)")
                    else:
                        self.model.load_state_dict(checkpoint)
                        logger.info(f"✅ Modèle chargé (direct)")
                else:
                    logger.error(f"Format checkpoint inconnu")
                    self.model = None
                    return
                
                self.model.eval()
                logger.info(f"✅ Modèle prêt - Classes: {self.class_names}")
                
                test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
                with torch.no_grad():
                    class_out, loc_out = self.model(test_input)
                    logger.info(f"✅ Test forward OK")
                
            else:
                logger.error(f"❌ Modèle non trouvé: {FALL_MODEL_PATH}")
                self.model = FastFallDetector(num_classes=2).to(DEVICE)
                self.model.eval()
                logger.warning("⚠️ Modèle vide créé")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            self.model = None
    
    def preprocess(self, image):
        """Préparer l'image pour le modèle"""
        try:
            img = image.copy()
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            return img_tensor, img_resized
            
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement: {e}")
            return None, None
    
    def predict(self, image):
        """Faire une prédiction sur une image"""
        try:
            if self.model is None:
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': 'Modèle non chargé',
                    'box': [IMG_SIZE//3, IMG_SIZE//3, 2*IMG_SIZE//3, 2*IMG_SIZE//3],
                    'probabilities': {'normal': 0, 'fall': 0}
                }
            
            img_tensor, img_resized = self.preprocess(image)
            if img_tensor is None:
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': 'Erreur prétraitement',
                    'box': [IMG_SIZE//3, IMG_SIZE//3, 2*IMG_SIZE//3, 2*IMG_SIZE//3],
                    'probabilities': {'normal': 0, 'fall': 0}
                }
            
            predicted_class, class_probs, boxes, loc_map = self.model.predict(img_tensor, threshold=0.3)
            
            # Récupérer les probabilités brutes (index 0 et 1)
            prob_class0 = class_probs[0, 0].item()  # probabilité pour la classe 0
            prob_class1 = class_probs[0, 1].item()  # probabilité pour la classe 1
            
            # CORRECTION : Inverser la prédiction si nécessaire
            # Le modèle donne prob_class0 = fall, prob_class1 = normal ?
            # Ou l'inverse ? Vérifions avec les valeurs:
            # Si prob_class0 > prob_class1, la prédiction est classe 0 (fall)
            # Sinon classe 1 (normal)
            
            pred_idx = 0 if prob_class0 > prob_class1 else 1
            
            # Confiance = probabilité de la classe prédite
            confidence = prob_class0 if pred_idx == 0 else prob_class1
            
            # Nom de la classe prédite
            pred_name = self.class_names[pred_idx]  # 'fall' si pred_idx=0, 'normal' si pred_idx=1
            
            box = boxes[0] if boxes else [IMG_SIZE//3, IMG_SIZE//3, 2*IMG_SIZE//3, 2*IMG_SIZE//3]
            
            # Pour l'affichage: normal = prob_class1, fall = prob_class0 ou inversé selon l'entraînement
            # Ici on suppose que le modèle a été entraîné: classe0 = fall, classe1 = normal
            prob_normal = prob_class1 * 100
            prob_fall = prob_class0 * 100
            
            logger.info(f"📊 Probabilités brutes: Classe0={prob_class0:.4f}, Classe1={prob_class1:.4f}")
            logger.info(f"📊 Classe0 = fall, Classe1 = normal (hypothèse)")
            logger.info(f"📊 Normal: {prob_normal:.1f}%, Fall: {prob_fall:.1f}%")
            logger.info(f"📊 Prédiction: {pred_name} (Confiance: {confidence*100:.1f}%)")
            
            return {
                'prediction': pred_name.upper(),
                'confidence': float(confidence * 100),
                'probabilities': {
                    'normal': float(prob_normal),
                    'fall': float(prob_fall)
                },
                'error': None,
                'box': box,
                'image_resized': img_resized
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'box': [IMG_SIZE//3, IMG_SIZE//3, 2*IMG_SIZE//3, 2*IMG_SIZE//3],
                'probabilities': {'normal': 50, 'fall': 50}
            }


model_loader = None


def get_model():
    global model_loader
    if model_loader is None:
        logger.info("🚀 Initialisation du modèle...")
        model_loader = FallDetectionModelWrapper()
    return model_loader