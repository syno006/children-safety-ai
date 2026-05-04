# blessures/apps.py
from django.apps import AppConfig


class BlessuresConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'blessures_app'
    
    def ready(self):
        # Précharge les modèles au démarrage
        from blessures_app.model_loader import model_loader
        # Force le chargement
        _ = model_loader