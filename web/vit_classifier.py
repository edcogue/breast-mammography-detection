from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import pipeline
import sys

id2label = {0: 'B', 1: 'M'}
label2id = {'B': 0, 'M': 1}

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k")

model = AutoModelForImageClassification.from_pretrained(
    "./models/vit_classifier/vit_class_google_vit-base-patch16-224-in21k_27052023_232704",
    num_labels=len(id2label.keys()),
    id2label=id2label,
    label2id=label2id,
).to("cpu")

classifier = pipeline("image-classification", model=model,
                      image_processor=image_processor)


def classify_pathology(image):
    results = classifier(image)
    return max(results, key=lambda x: x["score"])
