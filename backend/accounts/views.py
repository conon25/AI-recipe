from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import JsonResponse
from IPython import embed
from rest_framework.decorators import api_view
from rest_framework.response import Response
# Create your views here.
@api_view(['POST'])
def user_login(request):
    if request.user.is_authenticated:
        message = {
            'code' : 200
        }
        return JsonResponse(message)
    message = {
        'message' : 'hi'
    }
    return JsonResponse(message)