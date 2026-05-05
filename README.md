# 🏠 Shelter Safety AI

A Django web application integrating multiple AI models to ensure 
a safe and stable daily life for children in shelters.

Developed as part of the **3IA Program — Challenge-Based Learning Project** 
at [Esprit School of Engineering](https://esprit.tn).

---

## Overview

**Big Idea:** Child Safety  
**Essential Question:** How can we ensure that children stay safe every day in shelters?  
**Challenge:** Provide a safe and stable daily life for children in shelters.

This platform combines 7 AI objectives deployed through a unified Django 
backend, providing real-time risk detection and alerting for shelter staff.

---

## Features

- **Sanitation risk detection** — Identifies cockroaches, trash, and wet 
  surfaces using YOLO11s (mAP50: 0.819)
- **Structural damage assessment** — Detects spalling and efflorescence
- **Overcrowding prediction** — Estimates occupancy and generates density heatmaps
- **Accident hazard detection** — Identifies dangerous objects and missing 
  safety equipment
- **Behavioral risk detection** — Classifies risky child behaviors and 
  computes risk scores
- **Allergen detection** — Identifies allergenic substances in food ingredients
- **Meal nutrition analysis** — Analyzes meals for nutritional balance

---

## Tech Stack

### Backend
- Python 3.12
- Django
- Django REST Framework

### AI Models
- PyTorch 2.10
- Ultralytics YOLO11s
- OpenCV

### Explainable AI
- LIME
- Occlusion Saliency Maps

### Deployment
- Heroku (via GitHub Education)

---

## Directory Structure
## Getting Started

```bash
git clone https://github.com/your-username/shelter-safety-ai
cd shelter-safety-ai
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Acknowledgments

Developed under the guidance of professors at **Esprit School of Engineering**.  
CBL Project — 3IA Program 2025/2026.
