from django.shortcuts import render
from ultralytics import YOLO
from pathlib import Path
import csv

# ============================================================
# CHARGER MODELE
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "allergen_app" / "model" / "allergen_yolov8s_best.pt"

model = YOLO(str(MODEL_PATH))

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

        # ============================================================
        # YOLO PREDICTION
        # ============================================================
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