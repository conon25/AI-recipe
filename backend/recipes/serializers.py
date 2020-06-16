from rest_framework import serializers
from .models import *
class RecipeBasicInfoSerializer(serializers.ModelSerializer):
  class Meta:
    model = RecipeBasicInfo
    fields = '__all__'

class RecipeMaterialInfoSerializer(serializers.ModelSerializer):
  class Meta:
    model = RecipeMaterialInfo
    # fields = ('material_number', 'material_name',)
    fields = '__all__'
    
class RecipeProcessInfoSerializer(serializers.ModelSerializer):
  class Meta:
    model = RecipeProcessInfo
    fields = '__all__'