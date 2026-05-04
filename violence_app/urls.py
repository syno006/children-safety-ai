from django.urls import path
from . import views

urlpatterns = [
    path('',                views.index,        name='index'),
    path('upload/',         views.upload_view,  name='upload'),
    path('analyze/',        views.analyze,      name='analyze'),
    path('result/<int:pk>/',views.result_view,  name='result'),
    path('history/',        views.history_view, name='history'),
    path('api/analyze/',    views.api_analyze,  name='api-analyze'),
    path('api/analyze-frame/', views.api_analyze_frame, name='api-analyze-frame'),
    path('api/metrics/',    views.get_metrics,  name='api-metrics'),
    path('detection/', views.detection_view, name='detection'),
]