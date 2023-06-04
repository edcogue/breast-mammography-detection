import numpy as np
import albumentations as albu
import json
import cv2
from datetime import datetime

import sys
sys.path.append('../preprocess/data_loader')
from data_loader import DataLoader
import torch
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, ClassLabel
from PIL import ImageDraw
from PIL import Image as ImagePil
from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

# For evaluate
import evaluate
from tqdm import tqdm
import torchvision
import os

loader = DataLoader(input_shape=1333)

features = Features({'image_id': Value(dtype='int64', id=None),
                     'image': Image(decode=True, id=None),
                     'width': Value(dtype='int32', id=None),
                     'height': Value(dtype='int32', id=None),
                     'objects': Sequence(feature={'id': Value(dtype='int64', id=None),
                                                  'area': Value(dtype='int64', id=None),
                                                  'bbox': Sequence(feature=Value(dtype='float32', id=None),
                                                                   length=4,
                                                                   id=None),
                                                  'category': ClassLabel(names=["Abnormality"], id=None)
                                                  },
                                         length=-1,
                                         id=None)
                     })


dataset = Dataset.from_generator(loader.object_detection_generator(target_library="hugging_face", ),
                                 features=features,
                                 cache_dir="./dataset_detection_cache",
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

categories = dataset["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

print(id2label)
print(label2id)

checkpoint = "facebook/detr-resnet-50"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    #encoding = batch
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = loader.transform(image, objects["bbox"], objects["category"])

        area.append([w*h for _,_,w,h in out["bboxes"]])
        #area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
        

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]
    return image_processor(images=images, annotations=targets, return_tensors="pt")

train_ds = dataset["train"].with_transform(transform_aug_ann)
val_ds = dataset["valid"].with_transform(transform_aug_ann)

model = AutoModelForObjectDetection.from_pretrained(
    "./detr_facebook_detr-resnet-50_04062023_223806",
    ignore_mismatched_sizes=True,
)


# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["bbox"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


# Save images and annotations into the files torchvision.datasets.CocoDetection expects
def save_cppe5_annotation_file_images(cppe5):
    output_json = {}
    path_output_cppe5 = f"{os.getcwd()}/evaluate_files/"

    if not os.path.exists(path_output_cppe5):
        os.makedirs(path_output_cppe5)

    path_anno = os.path.join(path_output_cppe5, "evaluate.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    
    total=len(cppe5)
    for i, example in enumerate(cppe5):
        path_img = os.path.join(path_output_cppe5, f"{example['image_id']}.png")
        # Apply CLAHE and default transformations
        out = loader.transform(np.array(example["image"]), example["objects"]["bbox"], example["objects"]["category"])
        
        cv2.imwrite(path_img,out["image"])
        example["objects"]["bbox"]=out["bboxes"]
        ann = val_formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": out["image"].shape[0],
                "height": out["image"].shape[1],
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    return path_output_cppe5, path_anno

path_output_cppe5, path_anno = save_cppe5_annotation_file_images(dataset["test"])

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}


im_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

test_ds_coco_format = CocoDetection(path_output_cppe5, im_processor, path_anno)


module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
val_dataloader = torch.utils.data.DataLoader(
    test_ds_coco_format, batch_size=4, shuffle=False, collate_fn=collate_fn
)

with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]

        labels = [
            {k: v for k, v in t.items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

        module.add(prediction=results, reference=labels)
        del batch

results = module.compute()
print(results)