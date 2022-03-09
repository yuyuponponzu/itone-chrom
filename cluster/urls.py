from django.contrib import admin
from django.urls import path, include
from cluster import views

app_name = 'cluster'
urlpatterns = [
    path('cluster', views.cluster),
    path('segment', views.segment)
]