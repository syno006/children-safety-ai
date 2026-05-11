from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'core'

urlpatterns = [
    # Page de login comme page d'accueil
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('home/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('logout/', auth_views.LogoutView.as_view(next_page='core:login'), name='logout'),
]