import cv2
import requests
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def histgram(request):
    file_obj = request.data['file'].read()
    img = uploaded_file_to_cv(file_obj)
    hist = cv_to_hist(img)
    return Response(hist, status=200)

def uploaded_file_to_cv(file_obj):
    img_array = np.asarray(bytearray(file_obj), dtype=np.uint8)
    return cv2.imdecode(img_array, 1)

def cv_to_hist(img):
    histgram = []
    cvt_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i, channel in enumerate(["r", "g", "b"]):
        histgram.append(cv2.calcHist([cvt_color], [i], None, [256], [0, 256]))
    return histgram