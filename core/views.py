from django.shortcuts import render


def home(request):
    simulation = {
        'scene_video': '/media/uploads/0a250dd19dd24120a41525b045547875.mp4',
        'scene_poster': '/static/images/simulation-scene.jpg',
        'scene_title': 'Refuge — scene surveillance en direct',
        'scene_caption': 'Video surveillance partagee • Couloir • Zone de vie • Traitement temps reel',
        'scene_description': 'Flux video traite par tous les modeles d\'IA de ShelterCare pour detecter en temps reel les risques de securite pour les enfants.',
        'items': [
            {
                'label': 'Objets dangereux',
                'detail': 'Couteau découvert près d’une zone de préparation alimentaire. Modèles d’objets dangereux + incendie.',
                'badge': 'danger',
                'badge_text': 'Alerte critique',
            },
            {
                'label': 'Chute potentielle',
                'detail': 'Posture instable détectée dans le couloir. Modèle de détection de chutes.',
                'badge': 'warning',
                'badge_text': 'Risque élevé',
            },
            {
                'label': 'Risques sanitaires',
                'detail': 'Déchets et sol humide identifiés dans la cuisine. Modèle sanitation.',
                'badge': 'warning',
                'badge_text': 'Intervention requise',
            },
            {
                'label': 'Allergènes',
                'detail': 'Trace d’arachide détectée sur une table proche d’enfants allergiques. Modèle allergène.',
                'badge': 'danger',
                'badge_text': 'Attention immédiate',
            },
            {
                'label': 'Structure',
                'detail': 'Dégradation détectée sur un mur porteur. Modèle structural.',
                'badge': 'warning',
                'badge_text': 'Suivi recommandé',
            },
        ],
        'summary': 'ShelterCare unifie ces 5 modeles pour protection 360 degres en direct. Tous les flux detectent simultanement sur cette video surveillance.',
    }
    return render(request, 'core/home.html', {'simulation': simulation})


def about(request):
    return render(request, 'core/about.html')