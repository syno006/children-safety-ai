"""
structural_app/urls.py
"""
from django.urls import path
from . import views

app_name = "structural_app"

urlpatterns = [
    path("",           views.home,      name="home"),
    path("detection/", views.detection, name="detection"),
]
