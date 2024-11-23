from django.urls import path
from .views import AnalyzeCsvView

urlpatterns = [
    path('analyze/', AnalyzeCsvView.as_view(), name='analyze'),
]
