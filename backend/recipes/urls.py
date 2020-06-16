from django.urls import path
from . import views
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
   openapi.Info(
      title="AMT2",
      default_version='v1',
      description="AMT2",
      license=openapi.License(name="License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)

app_name = 'recipes'
urlpatterns = [
  path('basicinfo/', views.basicinfo, name='basicinfo'),
  path('basicinfo/<int:dish_pk>/', views.get_dish_by_id),
  path('materialinfo/<int:basic_pk>/', views.materialinfo),
  path('processinfo/<int:basic_pk>/', views.processinfo),
#   path('materialcheck/', views.materialcheck),
  path('image_upload/', views.image_upload),
  path('get_dishes/', views.get_dishes),
  path('mask_rcnn/', views.mask_rcnn),
#   path('image_test/', views.image_test),
  path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='swagger'),
]