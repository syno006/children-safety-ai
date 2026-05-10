from django.shortcuts import render
from pathlib import Path
import csv

# ============================================================
# CHARGER MODELE
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "allergen_app" / "model" / "allergen_yolov8s_best.pt"
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[allergen_app] WARNING – ultralytics is not installed.")
        return None

    if MODEL_PATH.exists():
        try:
            _model = YOLO(str(MODEL_PATH))
            print(f"[allergen_app] Model loaded from {MODEL_PATH}")
        except Exception as exc:
            print(f"[allergen_app] ERROR loading model: {exc}")
            _model = None
    else:
        print(f"[allergen_app] WARNING – model not found at {MODEL_PATH}")

    return _model

<<<<<<< HEAD
_model = None
_model_load_error = None


def get_model():
    global _model, _model_load_error
    if _model is not None or _model_load_error is not None:
        return _model

    if not MODEL_PATH.exists():
        _model_load_error = f"Model file not found: {MODEL_PATH}"
        return None

    try:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    except Exception as exc:
        _model_load_error = str(exc)
        _model = None

    return _model

=======
>>>>>>> d10794b (update django project and ai models)

# ============================================================
# LIRE CSV ENFANTS
# ============================================================
def load_enfants():
    enfants = []
    csv_path = BASE_DIR / "allergen_app" / "model" / "enfants.csv"

    with open(csv_path, newline='', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            enfants.append({
                "nom": row["nom"],
                "allergies": [a.strip().lower() for a in row["allergies"].split(";")]
            })

    return enfants


# ============================================================
# VIEW PRINCIPALE
# ============================================================
def detect_ui(request):
    detections = []
    alertes = []

    enfants = load_enfants()
    enfant = enfants[0]  # ⚠️ test (tu peux remplacer plus tard)

    if request.method == "POST" and request.FILES.get("image"):

        image = request.FILES["image"]

        img_path = "temp.jpg"
        with open(img_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        model = get_model()
        if model is None:
            return render(request, "allergies/upload.html", {
                "detections": detections,
                "alertes": ["⚠️ Modèle allergène introuvable. Vérifiez le fichier de poids ou installez le modèle."],
            })

        # ============================================================
        # YOLO PREDICTION
        # ============================================================
        model = get_model()
        if model is None:
            alertes.append(_model_load_error or "Le modèle n'a pas pu être chargé.")
        else:
            results = model.predict(img_path, conf=0.25)

            # ============================================================
            # 🔥 GARDER UNIQUEMENT LA MEILLEURE CONFIDENCE PAR CLASSE
            # ============================================================
            best_detections = {}

            for r in results:
                for box in r.boxes:
                    classe = model.names[int(box.cls)].lower()
                    conf = float(box.conf)

                    # garder seulement la meilleure détection
                    if classe not in best_detections or conf > best_detections[classe]:
                        best_detections[classe] = conf

            # ============================================================
            # FORMAT FINAL DES DETECTIONS
            # ============================================================
            detections = [
                {
                    "class": cls,
                    "confidence": round(conf, 2)
                }
                for cls, conf in sorted(best_detections.items(), key=lambda x: x[1], reverse=True)
            ]

            # ============================================================
            # ALERTES (UNE SEULE FOIS PAR ALLERGIE)
            # ============================================================
            alertes = []

            for cls in best_detections:
                if cls in enfant["allergies"]:
                    alertes.append(f"⚠️ {enfant['nom']} allergique à {cls}")

    return render(request, "allergies/upload.html", {
        "detections": detections,
        "alertes": alertes
    })