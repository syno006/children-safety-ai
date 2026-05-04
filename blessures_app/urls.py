# blessures/urls.py
from django.urls import path
from . import views

app_name = 'blessures'

urlpatterns = [
    path('', views.home, name='home'),
    path('classification/', views.classification_view, name='classification'),
    path('classification-xai/', views.classification_xai_view, name='classification_xai'),
    path('detection/', views.detection_view, name='detection'),
    path('segmentation/', views.segmentation_view, name='segmentation'),
    path('segmentation-roi/', views.segmentation_roi_view, name='segmentation_roi'),
    path('full-pipeline/', views.full_pipeline_view, name='full_pipeline'),
    path('video/', views.video_analysis_view, name='video_analysis'),
path('video/download/<str:filename>/', views.download_video, name='download_video'),
]