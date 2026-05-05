from django.urls import path
from . import views

app_name = 'fall_detection'

urlpatterns = [
    path('', views.home, name='home'),                      # Page d'accueil
    path('yolo/', views.yolo_detection, name='yolo'),       # Page YOLO
    path('classification/', views.classification_view, name='classification'),  # Page CNN
]