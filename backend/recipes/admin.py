from django.contrib import admin
from .models import *
# Register your models here.

class RecipeBasicInfoAdmin(admin.ModelAdmin):
    list_display = ('pk', 'basic_code', 'basic_name', 'basic_intro', 'basic_typecode', 'basic_type', 'basic_imgurl')
    ordering = ('pk',)
admin.site.register(RecipeBasicInfo, RecipeBasicInfoAdmin)

class RecipeMaterialAdmin(admin.ModelAdmin):
    list_display = ('pk', 'material_code', 'material_number', 'material_name')
    ordering = ('pk',)
admin.site.register(RecipeMaterialInfo, RecipeMaterialAdmin)

class RecipeProcessAdmin(admin.ModelAdmin):
    list_display = ('pk', 'process_code', 'process_order', 'process_info')
    ordering = ('pk',)
admin.site.register(RecipeProcessInfo, RecipeProcessAdmin)