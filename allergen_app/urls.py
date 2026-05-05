from django.urls import path
from .views import detect_ui

app_name = "allergen_app" 
urlpatterns = [
    path("detect/", detect_ui,name="detect"),
]