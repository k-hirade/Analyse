from django.urls import path
from .views import search_tweets
from .views import PushDataList

urlpatterns = [
    path('result/', search_tweets),
    path('push', PushDataList.as_view(), name='push'),
]

