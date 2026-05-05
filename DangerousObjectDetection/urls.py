from django.urls import path
from . import views

app_name = 'dangerous'

urlpatterns = [
    path('', views.index, name='index'),
    path('sources/create/', views.create_source, name='create-source'),
    path('sources/<int:source_id>/layout/', views.update_layout, name='update-layout'),
    path('sources/<int:source_id>/models/', views.update_models, name='update-models'),
    path('sources/<int:source_id>/camera/', views.update_camera_config, name='update-camera'),
    path('sources/<int:source_id>/run/', views.run_source, name='run-source'),
    path('sources/<int:source_id>/video-live/', views.run_video_live, name='run-video-live'),
    path('sources/<int:source_id>/frame/', views.run_frame, name='run-frame'),
    path('sources/<int:source_id>/stream/', views.run_stream, name='run-stream'),
    path('sources/<int:source_id>/delete/', views.delete_source, name='delete-source'),
]
