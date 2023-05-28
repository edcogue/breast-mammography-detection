import sys
from PIL import ImageDraw, ImageFont
from PIL import Image as ImagePil
import numpy as np
import torch

from yolo_classifier import *

from ultralytics import YOLO

model = YOLO('./models/yolo_object_detector/best.pt')

font_size = 50
font = ImageFont.truetype("./fonts/arial.ttf", font_size)


def detect_abnormalities(image):
    image = ImagePil.fromarray(image).convert("RGB")
    results = model(image)[0]
    draw = ImageDraw.Draw(image)
    for box in results.boxes.data.tolist():
        x, y, w, h, _, _ = tuple(box)
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        results = classify_pathology(image.crop((x, y, w, h)))
        color_type = classify_type(image.crop((x, y, w, h)))
        color_text = (0, 255, 0, 255) if results["label"] == "B" else (
            255, 0, 0, 255)
        draw.rectangle((x, y, w, h), outline=color_type, width=8)
        draw.text(get_text_position(x, y, image),
                  f"{results['label'].upper()} {float(results['score']):.2f}", font=font, fill=color_text, stroke_width=1)
    return image


def get_text_position(x, y, image):
    w = image.width
    h = image.height
    if (y - font_size) < 0:
        y = y + font_size
    else:
        y = y - font_size
    return x, y
