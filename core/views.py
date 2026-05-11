from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import datetime

def login_view(request):
    # Si l'utilisateur est déjà connecté, rediriger vers home
    if request.user.is_authenticated:
        return redirect('core:home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Bienvenue {user.username} ! 🌟 Ensemble, protégeons les enfants.')
            return redirect('core:home')
        else:
            messages.error(request, 'Email ou mot de passe incorrect. Veuillez réessayer.')
    
    return render(request, 'core/login.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('core:home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        full_name = request.POST.get('full_name')
        shelter_name = request.POST.get('shelter_name')
        
        # Validation des mots de passe
        if password1 != password2:
            messages.error(request, 'Les mots de passe ne correspondent pas.')
            return render(request, 'core/login.html', {'show_register': True})
        
        # Vérifier si l'utilisateur existe déjà
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Ce nom d\'utilisateur existe déjà.')
            return render(request, 'core/login.html', {'show_register': True})
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Cet email est déjà utilisé.')
            return render(request, 'core/login.html', {'show_register': True})
        
        # Créer l'utilisateur
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            first_name=full_name
        )
        
        # Vous pouvez ajouter des informations supplémentaires dans un profil utilisateur
        # user.profile.shelter_name = shelter_name
        # user.profile.save()
        
        # Afficher un message de succès et rediriger vers la page de connexion
        messages.success(request, f'Félicitations {full_name} ! 🎉 Votre compte a été créé avec succès. Veuillez vous connecter avec vos identifiants.')
        return redirect('core:login')
    
    return render(request, 'core/login.html', {'show_register': True})

@login_required(login_url='core:login')
def home(request):
    # Statistiques pour le dashboard
    context = {
        'user': request.user,
        'children_count': 234,
        'total_children': 142,
        'alerts_today': 3,
        'staff_count': 28,
        'current_time': datetime.now(),
        'simulation': {
            'scene_video': '/media/uploads/0a250dd19dd24120a41525b045547875.mp4',
            'scene_poster': '/static/images/simulation-scene.jpg',
            'scene_title': 'Refuge — scene surveillance en direct',
            'scene_caption': 'Video surveillance partagee • Couloir • Zone de vie • Traitement temps reel',
            'scene_description': 'Flux video traite par tous les modeles d\'IA de ShelterCare pour detecter en temps reel les risques de securite pour les enfants.',
            'items': [
                {
                    'label': 'Objets dangereux',
                    'detail': 'Couteau découvert près d\'une zone de préparation alimentaire. Modèles d\'objets dangereux + incendie.',
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
                    'detail': 'Trace d\'arachide détectée sur une table proche d\'enfants allergiques. Modèle allergène.',
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
    }
    return render(request, 'core/home.html', context)

def about(request):
    return render(request, 'core/about.html')