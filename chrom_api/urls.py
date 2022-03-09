from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('image/', include('analysis.urls')),
    path('image/', include('cluster.urls'))
]