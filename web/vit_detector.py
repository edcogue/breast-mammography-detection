from PIL import ImageDraw, ImageFont
from PIL import Image as ImagePil
import numpy as np
import torch

from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor

from vit_classifier import *
from yolo_classifier import classify_type


id2label = {0: 'Abnormality'}
label2id = {'Abnormality': 0}

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = AutoModelForObjectDetection.from_pretrained(
    "./models/vit_object_detection/vit__facebook_detr-resnet-50_20052023_172920",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to("cpu")

font_size = 50
font = ImageFont.truetype("./fonts/arial.ttf", font_size)


def detect_abnormalities(image):
    image = ImagePil.fromarray(image).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.75)[0]
    draw = ImageDraw.Draw(image)
    for score, box in zip(results["scores"], results["boxes"]):
        x, y, w, h = tuple(box)
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
                  f"{results['label'].upper()} {int(float(results['score'])*100)}%", font=font, fill=color_text, stroke_width=1)
    return image


def get_text_position(x, y, image):
    w = image.width
    h = image.height
    if (y - font_size) < 0:
        y = y + font_size
    else:
        y = y - font_size
    return x, y
