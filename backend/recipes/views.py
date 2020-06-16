from django.shortcuts import render, get_object_or_404, redirect
from IPython import embed
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import *
from .serializers import *
# Create your views here.
from django.http import HttpResponse

@api_view(['GET'])
def basicinfo(request):
  recipebasic = RecipeBasicInfo.objects.all()
  serializer = RecipeBasicInfoSerializer(recipebasic, many=True)
  # embed()
  return Response(serializer.data)

@api_view(['GET'])  
def get_dish_by_id(request, dish_pk):
  dish = RecipeBasicInfo.objects.filter(basic_code=dish_pk)
  serializer = RecipeBasicInfoSerializer(dish, many=True)
  return Response(serializer.data)

@api_view(['GET'])
def materialinfo(request, basic_pk):
  materials = RecipeMaterialInfo.objects.filter(material_code=basic_pk)
  # for i in materials:
  #   print(i.material_name)

  serializer = RecipeMaterialInfoSerializer(materials, many=True)
  # embed()
  return Response(serializer.data)

@api_view(['GET'])
def processinfo(request, basic_pk):
  process = RecipeProcessInfo.objects.filter(process_code=basic_pk)
  process = process.order_by('process_order')
  serializer = RecipeProcessInfoSerializer(process, many=True)
  # embed()
  return Response(serializer.data)

# @api_view(['GET'])
# def materialcheck(request):
#   # 사진에서 인식한 재료
#   get_materials = ['베이컨', '양배추', '소고기', '대파', '쌀', '고구마 큰거', '올리브유', '소면', '물', '포도']
#   # 사용자가 선택할 조미료
#   get_cond = ['설탕', '소금']
#   all_materials = set(get_materials + get_cond)

#   all_dishes = RecipeBasicInfo.objects.all()
#   dish_list = []
#   for dish in all_dishes:
#     recipe_materials = RecipeMaterialInfo.objects.filter(material_code=dish.basic_code)
#     cnt = 0
#     for r_m in recipe_materials:
#       for a_m in all_materials:
#         if r_m.material_name == a_m:
#           cnt += 1

#     if len(recipe_materials) == cnt:
#       dish_list.append(dish.basic_code)
#   # embed()
#   dish_list = RecipeBasicInfo.objects.filter(basic_code__in=dish_list)
#   serializer = RecipeBasicInfoSerializer(dish_list, many=True)
#   return Response(serializer.data)

from .forms import *
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2


import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from IPython import embed
model = None
mlb = None

@api_view(['POST'])
def image_upload(request):
  global model, mlb
  img = request.FILES.get('file').read()
  img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
  
  # 재료 분석
  # labels_materials = detect.detect(img, model, mlb)
  image = cv2.resize(img, (299, 299))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  if model == None:
    model = load_model('multilabel.model')
    mlb = pickle.loads(open('mlb.pickle', "rb").read())
  proba = model.predict(image)[0]
  labels_materials = zip(mlb.classes_, proba)
  get_materials = []
  get_percentages = []
  # 기준 % 이상 재료만 담기
  for get_label, get_prob in labels_materials:
    # if get_prob * 100 > 10:
    get_materials.append(get_label)
    get_percentages.append(round(get_prob * 100, 4))
    print(get_label,':', get_prob*100)
  
  # 사진에서 인식한 재료
  # get_materials = ['계란','전분', '부침가루', '밀가루', '배추', '호박', '대파', '다짐육', '콩', '마늘']
  data = {
    'materials': get_materials,
    'percentages': get_percentages
    }
  return Response(data)

import json
@api_view(['POST'])
def get_dishes(request):
  # 사용자가 선택한 재료와 조미료
  # embed()
  get_materials = request.data.get('materials')
  get_cond = request.data.get('condiments')
  get_cond.append('물')
  print(get_materials)
  print(get_cond)
  # 사용자가 선택한 재료와 조미료(임시)
  # get_materials = ['계란','전분', '부침가루', '밀가루', '배추', '호박', '대파', '다짐육', '콩', '마늘']
  # get_cond = ['설탕', '소금', '후추', '참기름']
  all_materials = set(get_materials + get_cond)

  all_dishes = RecipeBasicInfo.objects.all()
  dish_list = []
  for dish in all_dishes:
    recipe_materials = RecipeMaterialInfo.objects.filter(material_code=dish.basic_code)
    cnt = 0
    for r_m in recipe_materials:
      for a_m in all_materials:
        if a_m == r_m.material_name:
          cnt += 1

    if len(recipe_materials) == cnt:
      dish_list.append(dish.basic_code)

  dish_list = RecipeBasicInfo.objects.filter(basic_code__in=dish_list)
  serializer = RecipeBasicInfoSerializer(dish_list, many=True)
  return Response(serializer.data)


import mrcnn.Inspect_Food_model as val
@api_view(['POST'])
def mask_rcnn(request):
  img = request.FILES.get('file').read()
  masked_image, masked_materials = val.distinct(img)
  # plt.imshow(masked_image)
  # plt.show()
  # embed()
  print(masked_materials)
  get_materials = []
  get_percentages = []
  material_dict = dict()
  for get_label, get_prob in masked_materials:
    if get_label not in material_dict:
      material_dict[get_label] = get_prob
    else:
      if material_dict[get_label] < get_prob:
        material_dict[get_label] = get_prob
  print(material_dict)

  for mat, per in material_dict.items():
    get_materials.append(mat)
    get_percentages.append(round(per * 100, 4))
  
  # mask rcnn은 없는 재료는 아예 식별하지 않기 때문에 따로 0%로 추가해준다.
  material_set = ['감자', '계란', '고추', '사과', '스팸', '양파']
  for mat in material_set:
    if mat not in get_materials:
      get_materials.append(mat)
      get_percentages.append(0)

  # 사진에서 인식한 재료
  # get_materials = ['계란','전분', '부침가루', '밀가루', '배추', '호박', '대파', '다짐육', '콩', '마늘']
  data = {
    'materials': get_materials,
    'percentages': get_percentages
    }
  # masked_image = masked_image.tobytes()

  # plt.imshow(masked_image)
  # plt.show()

  return Response(data)
  # return HttpResponse(masked_image, content_type="image/jpeg")

# import base64
# import matplotlib.image as mpimg
# @api_view(['POST'])
# def image_test(request):
#   img = request.FILES.get('file').read()
  
#   return HttpResponse(img, content_type="image/jpeg")