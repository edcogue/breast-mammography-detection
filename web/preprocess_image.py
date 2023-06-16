import sys
import numpy as np
import albumentations as albu
from n2v.models import N2V
import cv2

input_shape_vit = 1600

denoiser = N2V(None, "n2v_2D", basedir="/app/models/denoiser")
        
def resize_pad_and_clahe(image):
    image=np.array(image)
    transform = albu.Compose([
        albu.LongestMaxSize(max_size=input_shape_vit),
        albu.PadIfNeeded(
            min_height=input_shape_vit, min_width=input_shape_vit, border_mode=0, value=(0, 0, 0)),
        albu.CLAHE(clip_limit=(1, 10), p=1),
    ])
    
    return transform(
    image=image)["image"]

def denoise(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return denoiser.predict(grayImage.reshape((input_shape_vit, input_shape_vit)), axes="YX")
