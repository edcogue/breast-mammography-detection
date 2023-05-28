import sys
import numpy as np
import albumentations as albu
from n2v.models import N2V
import cv2

input_shape_vit = (1600,1600)

denoiser = N2V(None, "n2v_2D", basedir="/app/models/denoiser")

def _resize_correct_side(image):
        h_in = image.shape[0]
        w_in = image.shape[1]

        input_is_wider = h_in <= w_in
        output_is_wider = input_shape_vit[0] <= input_shape_vit[1]

        increase_width_first = abs(
            h_in/input_shape_vit[0] - 1.0) < abs(w_in/input_shape_vit[1] - 1.0)
        N = input_shape_vit[1] if increase_width_first else input_shape_vit[0]
        if input_is_wider != output_is_wider:
            return albu.LongestMaxSize(max_size=N)
        else:
            return albu.SmallestMaxSize(max_size=N)
        
def resize_pad_and_clahe(image):
    image=np.array(image)
    transform = albu.Compose([
        _resize_correct_side(image),
        albu.PadIfNeeded(
            min_height=input_shape_vit[0], min_width=input_shape_vit[1], border_mode=0, value=(0, 0, 0)),
        albu.CLAHE(clip_limit=(1, 10), p=1),
    ])
    
    return transform(
    image=image)["image"]

def denoise(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return denoiser.predict(grayImage.reshape(input_shape_vit), axes="YX")
