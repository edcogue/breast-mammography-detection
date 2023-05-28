from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import pipeline

from ultralytics import YOLO

model = YOLO('./models/yolo_classifier/best.pt')


def classify_pathology(image):
    result = model(image, imgsz=320)[0]
    result = [{"label": result.names[i], "score": p} for i,p in enumerate(result.probs.cpu().numpy())]
    return max(result, key=lambda x: x["score"])
