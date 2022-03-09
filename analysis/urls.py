from django.contrib import admin
from django.urls import path, include
from analysis import views

app_name = 'analysis'
urlpatterns = [
    path('histgram', views.histgram)
]