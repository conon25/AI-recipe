from django.urls import path
from . import views

app_name = 'accounts'
urlpatterns = [
  path('user_login/', views.user_login),
]