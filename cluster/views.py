import cv2
import requests
import torch
import numpy as np
import albumentations as albu
from typing import Dict, List, Optional, Tuple, Union
from people_segmentation.pre_trained_models import create_model
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from rest_framework.decorators import api_view
from rest_framework.response import Response

model = create_model("Unet_2020-07-20")
MAX_SIZE = 512
transform = albu.Compose(
    [albu.LongestMaxSize(max_size=MAX_SIZE), albu.Normalize(p=1)], p=1
)

@api_view(['GET'])
def cluster(request):
    if "num" in request.GET:
        amount_initial_centers = int(request.GET.get("num"))
    else :
        amount_initial_centers = 2
    file_obj = request.data['file'].read()
    img = uploaded_file_to_cv(file_obj)
    reshape_img = img.reshape((img.shape[0] * img.shape[1], 3))
    xmeans = get_xmeans(reshape_img,amount_initial_centers)
    res = {}
    res["hex"] = get_hexarr(xmeans, reshape_img)
    res["model"] = xmeans
    return Response(res, status=200)

@api_view(['GET'])
def segment(request):
    file_obj = request.data['file'].read()
    img = uploaded_file_to_cv(file_obj)
    res = segment_people(img)
    return Response(res, status=200)

def uploaded_file_to_cv(file_obj):
    img_array = np.asarray(bytearray(file_obj), dtype=np.uint8)
    return cv2.imdecode(img_array, 1)

def get_xmeans(reshape_img, amount_initial_centers):
    initial_centers = kmeans_plusplus_initializer(reshape_img, amount_initial_centers).initialize()
    # クラスタリングの実行
    xmeans_instance = xmeans(reshape_img, initial_centers=initial_centers, )
    xmeans_instance.process()
    return xmeans_instance

def get_hexarr(xmeans_instance, reshape_img):
    classes = len(xmeans_instance._xmeans__centers)
    predict = xmeans_instance.predict(reshape_img)
    centers = np.array(xmeans_instance._xmeans__centers).astype(int, copy=False)[:,0:3]

    color_hex_arr = []
    for ind,rgb_arr in enumerate(centers):
        color_hex_arr.append('#%02x%02x%02x' % tuple(rgb_arr))
    return color_hex_arr

def segment_people(img):
    original_height, original_width = img.shape[:2]
    image = transform(image=img)["image"]
    padded_image, pads = pad(image, factor=MAX_SIZE, border=cv2.BORDER_CONSTANT)
    x = torch.unsqueeze(tensor_from_rgb_image(padded_image), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_AREA)
    mask_3_channels = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return img * mask_3_channels

def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLECT_101) -> tuple:
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def unpad(image: np.array, pads: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops patch from the center so that sides are equal to pads.
    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    Returns: cropped image
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    return image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]

def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)