import numpy as np
import albumentations as albu
import json
import cv2
from datetime import datetime

import sys
sys.path.append('../preprocess/data_loader')
import matplotlib.pyplot as plt
from data_loader import DataLoader
import torch
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, ClassLabel
from PIL import ImageDraw
from PIL import Image as ImagePil
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback

# For evaluate
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomAffine
from transformers import pipeline
import evaluate
from tqdm import tqdm
import torchvision
import os

loader = DataLoader()

features = Features({
    'image': Image(decode=True, id=None),
    'label': ClassLabel(names=["B","M"], id=None)
})


dataset = Dataset.from_generator(loader.classification_generator(target_library="hugging_face",
                                                                 output_size=224),
                                 features=features,
                                 cache_dir="./dataset_class_cache",
                                 )

dataset=dataset.shuffle()

# 90% train, 10% test + validation
train_testvalid = dataset.train_test_split(test_size=0.3)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


print(dataset)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size, scale=(0.65, 1)),
                       RandomAffine(degrees=90,translate=(0.1,0.1),),
                       ToTensor(),
                       normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

ds = dataset.with_transform(transforms)

data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps",
                                                                [1, 2, 4, 8]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                [1,2,4,8]),
        "warmup_ratio": trial.suggest_float("warmup_ratio",0.1,0.5,step=0.05),
    }

def model_init(trial):     
    return AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

training_args = TrainingArguments(
    output_dir="hyper_params",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=None,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    tokenizer=image_processor,
    model_init=model_init,
    compute_metrics=compute_metrics,
)

trainer.train()

best_trial = trainer.hyperparameter_search(
    direction="maximize", #default metric is loss
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=15,
)

print(best_trial)

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

early_stop = EarlyStoppingCallback(10)

training_args = TrainingArguments(
    output_dir=f'/workspace/test_vit_class_{checkpoint.replace("/","_")}_{datetime.now().strftime("%d%m%Y_%H%M%S")}',
    num_train_epochs=100,
    learning_rate=best_trial.hyperparameters["learning_rate"],
    weight_decay=best_trial.hyperparameters["weight_decay"],
    warmup_ratio=best_trial.hyperparameters["warmup_ratio"],
    gradient_accumulation_steps=best_trial.hyperparameters["gradient_accumulation_steps"],
    per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    per_device_eval_batch_size=8,
    metric_for_best_model="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    logging_steps=50,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)

trainer.train()

model_name=f'./vit_class_{checkpoint.replace("/","_")}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
model.save_pretrained(model_name)

metrics = trainer.evaluate(ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
print(metrics)