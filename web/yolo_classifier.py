from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import pipeline

from ultralytics import YOLO
import sys

model = YOLO('./models/yolo_classifier/best.pt')
model_type = YOLO('./models/yolo_type_classifier/best.pt')

type2color = {
    'Assymetry': (234, 247, 42,255),
    'Calcification':(255,0,0,255),
    'Cluster': (0,0,0,255),
    'Distortion': (161, 97, 19,255),
    'Mass': (0,0,255,255),
    'Spiculated Region': (156,8,158,255),
    'Unknown': (97,97,97,255)
}


def classify_pathology(image):
    result = model(image, imgsz=320)[0]
    result = [{"label": result.names[i], "score": p}
              for i, p in enumerate(result.probs.data.cpu().numpy())]
    return max(result, key=lambda x: x["score"])

def classify_type(image):
    result = model_type(image, imgsz=320)[0]
    return type2color[result.names[result.probs.top1]]
