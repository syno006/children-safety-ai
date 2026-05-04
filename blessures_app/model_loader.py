# blessures/model_loader.py (version complète avec segmentation améliorée)
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import cm
import base64

# Configuration
IMG_SIZE = 224
SEG_IMG_SIZE = 256
URGENT_THRESHOLD = 0.35
YOLO_CONF_THRESHOLD = 0.10

# Classes
URGENCY_CLASSES = ['Urgent', 'Non-Urgent', 'Incertain']
URGENCY_EMOJI = {0: '🔴 Urgent', 1: '🟢 Non-Urgent', 2: '🟡 Incertain'}
URGENCY_COLORS = {0: '#E74C3C', 1: '#27AE60', 2: '#F39C12'}

# Normalisation
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class InjuryClassifier(nn.Module):
    """Modèle de classification EfficientNet-B3"""
    def __init__(self, num_classes=3, dropout=0.45):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=False,
            num_classes=0, global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        self.gradients = None
        self.activations = None
        self.hook_handle = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        self.activations = features
        
        if self.training or x.requires_grad:
            if self.hook_handle is not None:
                self.hook_handle.remove()
            self.hook_handle = features.register_hook(self.activations_hook)
        
        x = self.backbone.global_pool(features)
        x = x.flatten(1)
        x = self.head(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations
    
    def clear_hooks(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.yolo_model = None
        self.seg_model = None
        self._load_models()
    
    def _load_models(self):
        """Charge les modèles"""
        models_path = Path(__file__).parent / 'models'
        
        classifier_path = models_path / 'best_model_finetune.pth'
        if classifier_path.exists():
            self.classifier = InjuryClassifier(num_classes=3).to(self.device)
            self.classifier.load_state_dict(torch.load(
                classifier_path, map_location=self.device, weights_only=False
            ))
            self.classifier.eval()
            print(f"✅ Classificateur chargé")
        else:
            print(f"⚠️ Classificateur non trouvé: {classifier_path}")
        
        yolo_path = models_path / 'yolo_best_model_v2.pt'
        if yolo_path.exists():
            self.yolo_model = YOLO(str(yolo_path))
            print(f"✅ YOLO chargé")
        else:
            print(f"⚠️ YOLO non trouvé: {yolo_path}")
        
        seg_path = models_path / 'best_seg.pth'
        if seg_path.exists():
            self.seg_model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=3,
                classes=1
            ).to(self.device)
            self.seg_model.load_state_dict(torch.load(
                seg_path, map_location=self.device, weights_only=False
            ))
            self.seg_model.eval()
            print(f"✅ U-Net chargé")
        else:
            print(f"⚠️ U-Net non trouvé: {seg_path}")
    
    def preprocess_image(self, img_array, target_size=IMG_SIZE):
        """Prétraitement pour la classification"""
        aug = A.Compose([
            A.Resize(target_size, target_size),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2()
        ])
        transformed = aug(image=img_array)
        return transformed['image'].unsqueeze(0).to(self.device)
    
    def preprocess_segmentation(self, img_array):
        """Prétraitement pour la segmentation"""
        aug = A.Compose([
            A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2()
        ])
        transformed = aug(image=img_array)
        return transformed['image'].unsqueeze(0).to(self.device)
    
    def predict_classification(self, img_array):
        """Classification simple"""
        if self.classifier is None:
            return self._simulate_classification(img_array)
        
        tensor = self.preprocess_image(img_array)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = 0 if probs[0] > URGENT_THRESHOLD else int(np.argmax(probs))
        
        return {
            'prediction': URGENCY_CLASSES[pred_idx],
            'emoji': URGENCY_EMOJI[pred_idx],
            'color': URGENCY_COLORS[pred_idx],
            'probabilities': {URGENCY_CLASSES[i]: float(probs[i] * 100) for i in range(3)},
            'confidence_score': float(probs[pred_idx] * 100)
        }
    
    def predict_with_combined_heatmap(self, img_array):
        """Combine Grad-CAM + Détection + Segmentation pour une meilleure localisation"""
        if self.classifier is None:
            result = self._simulate_classification(img_array)
            result['heatmap'] = {'b64': None, 'overlay_b64': None}
            return result
        
        tensor = self.preprocess_image(img_array)
        h, w = img_array.shape[:2]
        
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = 0 if probs[0] > URGENT_THRESHOLD else int(np.argmax(probs))
        
        result = {
            'prediction': URGENCY_CLASSES[pred_idx],
            'emoji': URGENCY_EMOJI[pred_idx],
            'color': URGENCY_COLORS[pred_idx],
            'probabilities': {URGENCY_CLASSES[i]: float(probs[i] * 100) for i in range(3)},
            'confidence_score': float(probs[pred_idx] * 100),
            'heatmap': {'b64': None, 'overlay_b64': None}
        }
        
        try:
            heatmap_gradcam = self._compute_gradcam(tensor, pred_idx, h, w)
            detection_result = self.predict_detection(img_array)
            heatmap_detection = self._create_detection_heatmap(detection_result, h, w)
            seg_result = self.predict_segmentation(img_array)
            heatmap_segmentation = self._create_segmentation_heatmap(seg_result, h, w) if seg_result['has_segmentation'] else None
            
            heatmap_combined = heatmap_gradcam * 0.5 + heatmap_detection * 0.3
            if heatmap_segmentation is not None:
                heatmap_combined = heatmap_combined * 0.6 + heatmap_segmentation * 0.4
            
            if heatmap_combined.max() > 0:
                heatmap_combined = heatmap_combined / heatmap_combined.max()
            
            heatmap_combined = cv2.GaussianBlur(heatmap_combined, (9, 9), 2)
            threshold = np.percentile(heatmap_combined, 70)
            heatmap_combined[heatmap_combined < threshold] = heatmap_combined[heatmap_combined < threshold] * 0.3
            
            if heatmap_combined.max() > 0:
                heatmap_combined = heatmap_combined / heatmap_combined.max()
            
            heatmap_colored = (cm.jet(heatmap_combined)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_array, 0.5, heatmap_colored, 0.5, 0)
            
            for box in detection_result.get('boxes', []):
                cv2.rectangle(overlay, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
            
            result['heatmap'] = {
                'b64': self._array_to_base64(heatmap_colored),
                'overlay_b64': self._array_to_base64(overlay)
            }
        except Exception as e:
            print(f"Erreur: {e}")
        
        return result
    
    def predict_with_xai(self, img_array):
        """Grad-CAM dédié pour la page XAI"""
        if self.classifier is None:
            result = self._simulate_classification(img_array)
            result['heatmap'] = {'b64': None, 'overlay_b64': None}
            return result
        
        tensor = self.preprocess_image(img_array)
        h, w = img_array.shape[:2]
        
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = 0 if probs[0] > URGENT_THRESHOLD else int(np.argmax(probs))
        
        result = {
            'prediction': URGENCY_CLASSES[pred_idx],
            'emoji': URGENCY_EMOJI[pred_idx],
            'color': URGENCY_COLORS[pred_idx],
            'probabilities': {URGENCY_CLASSES[i]: float(probs[i] * 100) for i in range(3)},
            'confidence_score': float(probs[pred_idx] * 100),
            'heatmap': {'b64': None, 'overlay_b64': None}
        }
        
        try:
            heatmap = self._compute_gradcam(tensor, pred_idx, h, w)
            heatmap_colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_array, 0.45, heatmap_colored, 0.55, 0)
            result['heatmap'] = {
                'b64': self._array_to_base64(heatmap_colored),
                'overlay_b64': self._array_to_base64(overlay)
            }
        except Exception as e:
            print(f"Erreur XAI: {e}")
        
        return result
    
    def _compute_gradcam(self, tensor, target_class, h, w):
        """Calcule la heatmap Grad-CAM"""
        self.classifier.clear_hooks()
        tensor_grad = tensor.clone().detach().requires_grad_(True)
        logits = self.classifier(tensor_grad)
        self.classifier.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)
        
        gradients = self.classifier.get_activations_gradient()
        activations = self.classifier.get_activations()
        
        if gradients is None or activations is None:
            return np.zeros((h, w), dtype=np.float32)
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i in range(activations.shape[1]):
            cam += pooled_gradients[i] * activations[0, i, :, :]
        
        heatmap = torch.relu(cam).cpu().detach().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 2)
        threshold = np.percentile(heatmap, 60)
        heatmap = np.where(heatmap > threshold, heatmap, heatmap * 0.2)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap
    
    def _create_detection_heatmap(self, detection_result, h, w):
        """Crée une heatmap à partir des détections YOLO"""
        heatmap = np.zeros((h, w))
        for box in detection_result.get('boxes', []):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            confidence = box['confidence'] / 100.0
            for i in range(y1, min(y2, h)):
                for j in range(x1, min(x2, w)):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    dist_x = abs(j - center_x) / ((x2 - x1) / 2) if (x2 - x1) > 0 else 1
                    dist_y = abs(i - center_y) / ((y2 - y1) / 2) if (y2 - y1) > 0 else 1
                    intensity = max(0, 1 - max(dist_x, dist_y)) * confidence
                    heatmap[i, j] = max(heatmap[i, j], intensity)
        return heatmap
    
    def _create_segmentation_heatmap(self, seg_result, h, w):
        """Crée une heatmap à partir de la segmentation"""
        if 'mask' not in seg_result:
            return None
        mask = seg_result['mask']
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        heatmap = mask.astype(np.float32) / 255.0
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3)
        return heatmap
    
    def predict_with_gradcam(self, img_array):
        """Grad-CAM standard (simplifié)"""
        if self.classifier is None:
            result = self._simulate_classification(img_array)
            result['heatmap'] = {'b64': None, 'overlay_b64': None}
            return result
        
        tensor = self.preprocess_image(img_array)
        h, w = img_array.shape[:2]
        
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = 0 if probs[0] > URGENT_THRESHOLD else int(np.argmax(probs))
        
        result = {
            'prediction': URGENCY_CLASSES[pred_idx],
            'emoji': URGENCY_EMOJI[pred_idx],
            'color': URGENCY_COLORS[pred_idx],
            'probabilities': {URGENCY_CLASSES[i]: float(probs[i] * 100) for i in range(3)},
            'confidence_score': float(probs[pred_idx] * 100),
            'heatmap': {'b64': None, 'overlay_b64': None}
        }
        
        try:
            heatmap = self._compute_gradcam(tensor, pred_idx, h, w)
            heatmap_colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_array, 0.5, heatmap_colored, 0.5, 0)
            result['heatmap'] = {
                'b64': self._array_to_base64(heatmap_colored),
                'overlay_b64': self._array_to_base64(overlay)
            }
        except Exception as e:
            print(f"Erreur: {e}")
        
        return result
    
    def _array_to_base64(self, img_array):
        """Convertit une image en base64"""
        try:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
        except:
            return None
    
    def predict_detection(self, img_array):
        """Détection YOLO"""
        if self.yolo_model is None:
            return self._simulate_detection(img_array)
        
        h, w = img_array.shape[:2]
        results = self.yolo_model(img_array, conf=YOLO_CONF_THRESHOLD, iou=0.5, verbose=False)
        
        boxes = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                width = x2 - x1
                height = y2 - y1
                conf = float(box.conf[0].cpu().numpy())
                if width * height > 100:
                    boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': width, 'height': height,
                        'confidence': conf * 100,
                        'area_percentage': (width * height / (h * w)) * 100
                    })
        
        return {
            'boxes': boxes,
            'count': len(boxes),
            'has_detections': len(boxes) > 0,
            'total_area': sum(b['area_percentage'] for b in boxes) if boxes else 0,
            'max_confidence': max([b['confidence'] for b in boxes]) if boxes else 0,
            'avg_confidence': np.mean([b['confidence'] for b in boxes]) if boxes else 0
        }
    
    def predict_detection_with_fallback(self, img_array):
        """Détection avec fallback"""
        result = self.predict_detection(img_array)
        
        if not result['has_detections']:
            h, w = img_array.shape[:2]
            box_w, box_h = int(w * 0.6), int(h * 0.6)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            result = {
                'boxes': [{'x1': x1, 'y1': y1, 'x2': x1+box_w, 'y2': y1+box_h,
                          'width': box_w, 'height': box_h, 'confidence': 70.0,
                          'area_percentage': (box_w * box_h / (h * w)) * 100}],
                'count': 1, 'has_detections': True,
                'total_area': (box_w * box_h / (h * w)) * 100,
                'max_confidence': 70.0, 'avg_confidence': 70.0,
                'method': 'default'
            }
        else:
            result['method'] = 'yolo'
        
        return result
    
    @torch.no_grad()
    def predict_segmentation(self, img_array):
        """
        Segmentation améliorée - ne couvre que la blessure avec seuillage robuste
        """
        if self.seg_model is None:
            return self._simulate_segmentation(img_array)
        
        h, w = img_array.shape[:2]
        tensor = self.preprocess_segmentation(img_array)
        logits = self.seg_model(tensor)
        mask = torch.sigmoid(logits[0, 0]).cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        
        # 1. Analyse des valeurs du masque
        mask_min, mask_max, mask_mean = mask.min(), mask.max(), mask.mean()
        print(f"🔍 Masque - min: {mask_min:.3f}, max: {mask_max:.3f}, mean: {mask_mean:.3f}")
        
        # 2. Seuillage adaptatif plus strict
        if mask_mean > 0.8:
            # Masque très uniforme et élevé - probablement du bruit
            threshold = min(0.95, mask_max * 0.9)
        elif mask_mean > 0.6:
            threshold = min(0.85, mask_max * 0.8)
        elif mask_mean > 0.4:
            threshold = 0.75
        elif mask_mean > 0.2:
            threshold = 0.65
        else:
            threshold = 0.55
        
        print(f"   Seuil utilisé: {threshold}")
        
        # 3. Binarisation
        mask_binary = (mask > threshold).astype(np.uint8) * 255
        
        # 4. Nettoyage morphologique très agressif
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        # Plusieurs passes de nettoyage
        for _ in range(2):
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_small)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_medium)
        
        # 5. Filtrage par taille des composants
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        # Garder uniquement les composants > 1% de l'image et < 50%
        min_area = h * w * 0.01  # 1% minimum
        max_area = h * w * 0.5   # 50% maximum
        
        mask_filtered = np.zeros_like(mask_binary)
        valid_components = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                mask_filtered[labels == i] = 255
                valid_components += 1
        
        if valid_components == 0:
            # Si aucun composant valide, essayer avec un seuil encore plus élevé
            print("⚠️ Aucun composant valide trouvé, seuil augmenté drastiquement")
            threshold = min(0.98, mask_max * 0.95)
            mask_binary = (mask > threshold).astype(np.uint8) * 255
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_large)
            mask_filtered = mask_binary
        
        mask_binary = mask_filtered
        
        # 6. Éliminer les très petites régions restantes
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_final = h * w * 0.005  # 0.5% de l'image minimum
        for contour in contours:
            if cv2.contourArea(contour) < min_area_final:
                cv2.drawContours(mask_binary, [contour], -1, 0, -1)
        
        # 7. Calcul de la surface
        area_pixels = np.sum(mask_binary > 0)
        area_percentage = (area_pixels / (h * w)) * 100
        
        print(f"📊 Surface segmentée: {area_percentage:.1f}% de l'image")
        
        # 8. Validation finale - si toujours trop grand, considérer comme échec
        if area_percentage > 70:
            print("🚫 Segmentation invalide - surface trop grande, retour à None")
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'blended': img_array.copy(),
                'area_percentage': 0.0,
                'area_pixels': 0,
                'has_segmentation': False
            }
        
        # 9. Création des visualisations
        overlay = img_array.copy()
        overlay[mask_binary > 0] = [255, 80, 80]
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        blended = cv2.addWeighted(img_array, 0.55, overlay, 0.45, 0)
        
        return {
            'mask': mask_binary,
            'blended': blended,
            'area_percentage': area_percentage,
            'area_pixels': int(area_pixels),
            'has_segmentation': area_percentage > 0.5 and area_percentage < 60
        }
    
    def predict_segmentation_on_roi(self, img_array, detection_boxes=None):
        """
        Segmentation UNIQUEMENT sur la région d'intérêt (ROI) - Version améliorée
        """
        h, w = img_array.shape[:2]
        
        if detection_boxes is None:
            detection_result = self.predict_detection_with_fallback(img_array)
            detection_boxes = detection_result.get('boxes', [])
        
        if not detection_boxes:
            print("⚠️ Aucune région d'intérêt trouvée")
            return None
        
        all_segmentations = []
        
        for i, box in enumerate(detection_boxes):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi = img_array[y1:y2, x1:x2].copy()
            
            if roi.size == 0:
                continue
            
            # Segmenter UNIQUEMENT la ROI
            roi_seg = self._segment_roi_improved(roi)
            
            if roi_seg is not None and np.sum(roi_seg > 0) > 100:
                if roi_seg.shape[:2] != (y2-y1, x2-x1):
                    roi_seg = cv2.resize(roi_seg, (x2-x1, y2-y1))
                
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = roi_seg
                
                kernel = np.ones((3, 3), np.uint8)
                full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
                full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
                
                overlay = img_array.copy()
                overlay[full_mask > 0] = [255, 80, 80]
                
                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                
                area_pixels = int(np.sum(full_mask > 0))
                area_percentage = (area_pixels / (h * w)) * 100
                roi_size = (y2-y1) * (x2-x1)
                roi_area_percentage = (area_pixels / roi_size) * 100 if roi_size > 0 else 0
                
                all_segmentations.append({
                    'box_id': i,
                    'box': box,
                    'mask': full_mask,
                    'overlay': overlay,
                    'blended': cv2.addWeighted(img_array, 0.55, overlay, 0.45, 0),
                    'area_pixels': area_pixels,
                    'area_percentage': area_percentage,
                    'roi_area_percentage': roi_area_percentage
                })
                print(f"✅ ROI {i}: segmentation trouvée - {roi_area_percentage:.1f}% de la ROI")
        
        return all_segmentations if all_segmentations else None
    
    def _segment_roi_improved(self, roi_img):
        """
        Segmentation d'une région d'intérêt - Version améliorée
        """
        if self.seg_model is None:
            return self._simulate_roi_segmentation(roi_img)
        
        try:
            h_roi, w_roi = roi_img.shape[:2]
            
            tensor = self.preprocess_segmentation(roi_img)
            
            with torch.no_grad():
                logits = self.seg_model(tensor)
                mask = torch.sigmoid(logits[0, 0]).cpu().numpy()
            
            mask_resized = cv2.resize(mask, (w_roi, h_roi))
            
            mask_mean = mask_resized.mean()
            print(f"   ROI segmentation - mask mean: {mask_mean:.3f}")
            
            if mask_mean > 0.5:
                threshold = 0.75
            elif mask_mean > 0.3:
                threshold = 0.65
            else:
                threshold = 0.55
            
            mask_binary = (mask_resized > threshold).astype(np.uint8) * 255
            
            kernel = np.ones((3, 3), np.uint8)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
            
            # Garder uniquement le plus grand contour
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area_ratio = cv2.contourArea(largest) / (h_roi * w_roi)
                
                if area_ratio > 0.6 and mask_mean > 0.4:
                    print(f"   ROI - contour trop grand ({area_ratio*100:.1f}%), seuil augmenté")
                    mask_binary = (mask_resized > 0.85).astype(np.uint8) * 255
                    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
                    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                
                mask_clean = np.zeros((h_roi, w_roi), dtype=np.uint8)
                cv2.drawContours(mask_clean, [largest], -1, 255, -1)
                mask_binary = mask_clean
            
            return mask_binary
            
        except Exception as e:
            print(f"Erreur segmentation ROI: {e}")
            return self._simulate_roi_segmentation(roi_img)
    
    def _simulate_roi_segmentation(self, roi_img):
        """Simulation de segmentation pour la ROI"""
        h, w = roi_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 3
        
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x)**2 + (Y - center_y)**2 <= radius**2).astype(np.uint8) * 255
        
        return mask
    
    def _simulate_classification(self, img_array):
        """Simulation classification"""
        return {
            'prediction': 'Non-Urgent',
            'emoji': '🟢 Non-Urgent',
            'color': '#27ae60',
            'probabilities': {'Urgent': 15.0, 'Non-Urgent': 75.0, 'Incertain': 10.0},
            'confidence_score': 75.0
        }
    
    def _simulate_detection(self, img_array):
        """Simulation détection"""
        h, w = img_array.shape[:2]
        box_w, box_h = int(w * 0.6), int(h * 0.6)
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        return {
            'boxes': [{'x1': x1, 'y1': y1, 'x2': x1+box_w, 'y2': y1+box_h,
                      'width': box_w, 'height': box_h, 'confidence': 85.0,
                      'area_percentage': (box_w * box_h / (h * w)) * 100}],
            'count': 1, 'has_detections': True,
            'total_area': (box_w * box_h / (h * w)) * 100,
            'max_confidence': 85.0, 'avg_confidence': 85.0
        }
    
    def _simulate_segmentation(self, img_array):
        """Simulation segmentation"""
        h, w = img_array.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 4
        Y, X = np.ogrid[:h, :w]
        mask = ((X - center_x)**2 + (Y - center_y)**2 <= radius**2).astype(np.uint8) * 255
        area_percentage = (np.sum(mask > 0) / (h * w)) * 100
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 80, 80]
        blended = cv2.addWeighted(img_array, 0.55, overlay, 0.45, 0)
        return {
            'mask': mask,
            'blended': blended,
            'area_percentage': area_percentage,
            'area_pixels': int(np.sum(mask > 0)),
            'has_segmentation': area_percentage > 1
        }


# Instance globale
model_loader = ModelLoader()