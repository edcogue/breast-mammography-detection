import numpy as np
import os
import json
import tensorflow as tf
import albumentations as albu
import cv2
from n2v.models import N2V
from PIL import Image
import sys
sys.path.append('/tf/code/vit-object-detection/detectron2')
try:
    from detectron2.structures import BoxMode
except:
    print("Detectron2 not imported")

    

INBREAST_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/INBreast"
MIAS_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/MIAS"
CBIS_DDSM_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/CBIS-DDSM"


class DataLoader():
    def __init__(self, inbreast_path=INBREAST_PATH, mias_path=MIAS_PATH, cbis_ddsm_path=CBIS_DDSM_PATH, input_shape=1600, denoiser_model_path="/kaggle/working/models", denoiser_name="n2v_2D"):
        self.inbreast_path = inbreast_path
        self.mias_path = mias_path
        self.cbis_ddsm_path = cbis_ddsm_path
        self.input_shape = input_shape

        with open(os.path.join(self.inbreast_path, "roi_images.json"), "r") as f:
            rois_inbreast = json.load(f)

        with open(os.path.join(self.cbis_ddsm_path, "roi_images.json"), "r") as f:
            rois_cbis_ddsm = json.load(f)

        with open(os.path.join(self.mias_path, "roi_images.json"), "r") as f:
            rois_mias = json.load(f)

        self.denoiser_model_path=denoiser_model_path
        self.denoiser_name=denoiser_name
        self.last_id=0

        self.length = len(rois_cbis_ddsm) + len(rois_inbreast) + len(rois_mias)

        self.length_rois = sum([len(rois) for dataset in [rois_inbreast, rois_cbis_ddsm, rois_mias] for k, rois in dataset.items()])

        self.datasets_rois = {
            "mias": rois_mias, 
            "inbreast": rois_inbreast, 
            "cbis_ddsm": rois_cbis_ddsm
        }

        self.dataset_to_dir = {
            "mias": self.mias_path,
            "inbreast": self.inbreast_path,
            "cbis_ddsm": self.cbis_ddsm_path
        }
        
        self.type2int= {
            "Mass": 0,
            "Unknown": 1,
            "Assymetry": 2,
            "Distortion": 3,
            "Spiculated Region": 4,
            "Calcification": 5,
            "Cluster": 6,
            "Assymetry": 7
        }
        
        self.int2type= {
            0: "Mass",
            1: "Unknown",
            2: "Assymetry",
            3: "Distortion",
            4: "Spiculated Region",
            5: "Calcification",
            6: "Cluster",
            7: "Assymetry"
        }
        
        self.n_types = len(self.int2type.keys())

    def _load_denoiser(self):
        self.model = N2V(None, self.denoiser_name, basedir=self.denoiser_model_path)

    def _get_images_paths(self, path):
        images_path = []
        for (filepath, _, files) in os.walk(path, topdown=False):
            for file in files:
                if file.endswith(".png") and "MASK" not in file:
                    images_path.append(os.path.join(filepath, file))
        return images_path

    def _check_type_abnormality(self, ab):
        return ab["type"] in ["Calcification", "Mass"]

    def _load_img_from_path_no_resize(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=1)
        return img
    
    def _to_hugging_face(self, image_id, img, boxes):
        response = {
            "image_id": image_id,
            "image": img,
            "width": img.size[0],
            "height": img.size[1],
            "objects": {
                "id": [self.last_id+i+1 for i in range(len(boxes))],
                "area": [ box[2]*box[3] for box in boxes ],
                "bbox": boxes,
                "category": ["Abnormality" for i in range(len(boxes))],
            }
        }
        self.last_id+=len(boxes)
        return response
    
    def _format_hugging_face(self, image_id, path, boxes):
        with Image.open(path) as img:
            return self._to_hugging_face(image_id, img, boxes)
        
    def _format_detectron(self, image_id, path, boxes):
        with Image.open(path) as img:
            response = {
                "file_name": path,
                "width": img.size[0],
                "height": img.size[1],
                "image_id": image_id,
                "annotations": [{
                    "bbox": box,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [[box[0], box[1],
                                     box[0]+box[2], box[1],
                                     box[0]+box[2], box[1]+box[3],
                                     box[0], box[1]+box[3],
                                     box[0], box[1],
                                    ]],
                    "category_id": 0,
                } for box in boxes]
            }
            return response
        
    def _format_common(self, path, boxes, denoise, file, return_filename):
        img = self._load_img_from_path_no_resize(path)
        img, boxes = self.preprocess_img_and_boxes(
            img, boxes)
        if denoise:
            img = self.model.predict(
                np.array(img).reshape(self.input_shape), axes="YX")
        return (img, boxes, file) if return_filename else (img, boxes)
    
    def _format_for(self, image_id, path, boxes, file, denoise, return_filename, target_library=None):
        if target_library == "hugging_face":
            return self._format_hugging_face(image_id, path, boxes)
        if target_library == "detectron2":
            return self._format_detectron(image_id, path, boxes)
        return self._format_common(path, boxes, denoise, file, return_filename)

    def object_detection_generator(self, return_filename=False, denoise=False, target_library=None):
        if denoise:
            self._load_denoiser()
        def data_generator():
            self.last_id=0
            image_id=0
            for dataset, rois in self.datasets_rois.items():
                for file, roi in rois.items():
                    roi = list(
                        filter(lambda r: self._check_type_abnormality(r), roi))
                    if not roi:
                        continue

                    path = os.path.join(
                        self.dataset_to_dir[dataset], file+".png")
                    
                    boxes = [[box["x"], box["y"], box["w"], box["h"]]
                                 for box in roi]
                    image_id+=1
                    response = self._format_for(image_id, path, boxes, file, denoise, return_filename, target_library)
                    yield response
                    

        return data_generator
    
    def _gen_hot_encode(self,value):
        hot_encode = np.zeros(self.n_types)
        hot_encode[value]= 1.0
        return hot_encode
    
    def _class_format_for(self, img, preprocessed_img, classify_types, box, for_display, output_size, target_library=None, types_as=None):
        if target_library=="hugging_face":
            return {"image": Image.fromarray(np.array(preprocessed_img).reshape(output_size)),
                               "label": box["pathology"]}
        if classify_types:
            if types_as == "int":
                return preprocessed_img, self.type2int[box["type"]]
            if types_as == "hot_encode":
                return preprocessed_img, self._gen_hot_encode(self.type2int[box["type"]])
            
            return preprocessed_img, box["type"]
        return (preprocessed_img, box, img) if for_display else (preprocessed_img, box["pathology"])

    def classification_generator(self, output_size = None, classify_types=False, for_display=False, denoise=False, target_library=None, types_as=None):
        if denoise:
            self._load_denoiser()
        if output_size is None:
            output_size = self.input_shape
        def data_generator():
            for dataset, rois in self.datasets_rois.items():
                for file, roi in rois.items():
                    if not classify_types:
                        roi = list(
                            filter(lambda r: self._check_type_abnormality(r), roi))
                    if not roi:
                        continue

                    path = os.path.join(
                        self.dataset_to_dir[dataset], file+".png")
                    img = self._load_img_from_path_no_resize(path)
                    for box in roi:
                        preprocessed_img = self._crop_image(
                            img, box, output_size, augmentation=target_library!="hugging_face")

                        if denoise:
                            preprocessed_img = self.model.predict(
                                np.array(preprocessed_img).reshape(self.input_shape), axes="YX")
                        response = self._class_format_for(img, preprocessed_img, classify_types, box, for_display, output_size, target_library, types_as)
                        yield response

        return data_generator
        
    def transform(self, image, bboxes, labels):
            transform = albu.Compose([
                albu.LongestMaxSize(max_size=self.input_shape),
                albu.PadIfNeeded(
                    min_height=self.input_shape, min_width=self.input_shape, border_mode=0, value=(0, 0, 0)),
                albu.CLAHE(clip_limit=(1, 10), p=1),
                # Add as many transformations as needed
                albu.Rotate(p=0.2, border_mode=cv2.BORDER_CONSTANT,),
                albu.HorizontalFlip(p=0.2),
                albu.VerticalFlip(p=0.2),
            ],
                bbox_params=albu.BboxParams(
                    format='coco', label_fields=["category"])
            )
            return transform(
            image=image, bboxes=bboxes, category=labels)

    def _apply_albumentations(self, image, bboxes):
        image = image.numpy().astype("uint8")

        transform = albu.Compose([
            albu.LongestMaxSize(max_size=self.input_shape),
            albu.PadIfNeeded(
                min_height=self.input_shape, min_width=self.input_shape, border_mode=0, value=(0, 0, 0)),
            albu.CLAHE(clip_limit=(1, 10), p=1),
            # Add as many transformations as needed
            albu.Rotate(p=0.2, border_mode=cv2.BORDER_CONSTANT,),
            albu.HorizontalFlip(p=0.2),
            albu.VerticalFlip(p=0.2),
        ],
            bbox_params=albu.BboxParams(
                format='coco', label_fields=["abnormality"])
        )

        transform_data = transform(
            image=image, bboxes=bboxes, abnormality=[1]*len(bboxes))
        new_img = transform_data["image"]
        new_boxes = transform_data["bboxes"]
        return new_img.astype("uint8"), new_boxes

    def _crop_image(self, image, box, output_size, augmentation=True):
        image = image.numpy().astype("uint8")
        
        if augmentation:
            transform = albu.Compose([
                albu.CLAHE(clip_limit=(1, 10), p=1),
                albu.Crop(x_min=int(box["x"]), y_min=int(box["y"]),
                          x_max=int(box["x"]+box["w"]), y_max=int(box["y"]+box["h"])),
                albu.LongestMaxSize(max_size=output_size),
                albu.PadIfNeeded(
                    min_height=output_size, min_width=output_size, border_mode=0, value=(0, 0, 0)),
                # Add as many transformations as needed
                albu.Rotate(p=0.2, border_mode=cv2.BORDER_CONSTANT,),
                albu.HorizontalFlip(p=0.2),
                albu.VerticalFlip(p=0.2),
            ])
        else:
            transform = albu.Compose([
                albu.CLAHE(clip_limit=(1, 10), p=1),
                albu.Crop(x_min=int(box["x"]), y_min=int(box["y"]),
                          x_max=int(box["x"]+box["w"]), y_max=int(box["y"]+box["h"])),
                albu.LongestMaxSize(max_size=output_size),
                albu.PadIfNeeded(
                    min_height=output_size, min_width=output_size, border_mode=0, value=(0, 0, 0)),
            ])

        transform_data = transform(image=image)
        new_img = transform_data["image"]
        return new_img.astype("uint8")

    def preprocess_img_and_boxes(self, image, boxes):
        image, boxes = self._apply_albumentations(image, boxes)
        return image, boxes
