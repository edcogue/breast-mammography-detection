import numpy as np
import os
import json
import tensorflow as tf
import albumentations as albu
import cv2
from n2v.models import N2V

INBREAST_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/INBreast"
MIAS_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/MIAS"
CBIS_DDSM_PATH = "/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies/CBIS-DDSM"


class DataLoader():
    def __init__(self, inbreast_path=INBREAST_PATH, mias_path=MIAS_PATH, cbis_ddsm_path=CBIS_DDSM_PATH, input_shape=(1600, 1600), denoising=False, denoiser_model_path="/kaggle/working/models", denoiser_name="n2v_2D"):
        self.inbreast_path = inbreast_path
        self.mias_path = mias_path
        self.cbis_ddsm_path = cbis_ddsm_path
        self.input_shape = input_shape
        self.enable_denoising = denoising

        if self.enable_denoising:
            self._load_denoiser(denoiser_model_path, denoiser_name)

        with open(os.path.join(self.inbreast_path, "roi_images.json"), "r") as f:
            rois_inbreast = json.load(f)

        with open(os.path.join(self.cbis_ddsm_path, "roi_images.json"), "r") as f:
            rois_cbis_ddsm = json.load(f)

        with open(os.path.join(self.mias_path, "roi_images.json"), "r") as f:
            rois_mias = json.load(f)

        

        self.length = len(rois_cbis_ddsm.keys()) + len(rois_inbreast.keys()) + len(rois_mias.keys())

        self.datasets_rois = {
            "mias": rois_mias, "inbreast": rois_inbreast, "cbis_ddsm": rois_cbis_ddsm}

        self.dataset_to_dir = {
            "mias": self.mias_path, "inbreast": self.inbreast_path, "cbis_ddsm": self.cbis_ddsm_path}

    def _load_denoiser(self, basedir, model_name):
        self.model = N2V(None, model_name, basedir=basedir)

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

    def object_detection_generator(self):
        def data_generator():
            for dataset, rois in self.datasets_rois.items():
                for file, roi in rois.items():
                    roi = list(
                        filter(lambda r: self._check_type_abnormality(r), roi))
                    if not roi:
                        continue

                    path = os.path.join(
                        self.dataset_to_dir[dataset], file+".png")
                    img = self._load_img_from_path_no_resize(path)

                    boxes = [[box["x"], box["y"], box["w"], box["h"]]
                             for box in roi]

                    preprocessed_img, preprocessed_boxes = self._preprocess_img_and_boxes(
                        img, boxes)

                    if self.enable_denoising:
                        preprocessed_img = self.model.predict(
                            np.array(preprocessed_img).reshape(self.input_shape), axes="YX")

                    yield preprocessed_img, preprocessed_boxes

        return data_generator

    def classification_generator(self, output_size = None, for_display=False):
        if output_size is None:
            output_size = self.input_shape
        def data_generator():
            for dataset, rois in self.datasets_rois.items():
                for file, roi in rois.items():
                    roi = list(
                        filter(lambda r: self._check_type_abnormality(r), roi))
                    if not roi:
                        continue

                    path = os.path.join(
                        self.dataset_to_dir[dataset], file+".png")
                    img = self._load_img_from_path_no_resize(path)
                    for box in roi:
                        preprocessed_img = self._crop_image(
                            img, box, output_size)

                        if self.enable_denoising:
                            preprocessed_img = self.model.predict(
                                np.array(preprocessed_img).reshape(self.input_shape), axes="YX")
                        yield preprocessed_img, box, img if for_display else preprocessed_img, box["pathology"]

        return data_generator

    def _resize_correct_side(self, image):
        h_in = image.shape[0]
        w_in = image.shape[1]

        input_is_wider = h_in <= w_in
        output_is_wider = self.input_shape[0] <= self.input_shape[1]

        increase_width_first = abs(
            h_in/self.input_shape[0] - 1.0) < abs(w_in/self.input_shape[1] - 1.0)
        N = self.input_shape[1] if increase_width_first else self.input_shape[0]
        if input_is_wider != output_is_wider:
            return albu.LongestMaxSize(max_size=N)
        else:
            return albu.SmallestMaxSize(max_size=N)

    def _apply_albumentations(self, image, bboxes):
        image = image.numpy().astype("uint8")

        transform = albu.Compose([
            self._resize_correct_side(image),
            albu.PadIfNeeded(
                min_height=self.input_shape[0], min_width=self.input_shape[1], border_mode=0, value=(0, 0, 0)),
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

    def _crop_image(self, image, box, output_size):
        image = image.numpy().astype("uint8")

        transform = albu.Compose([
            albu.CLAHE(clip_limit=(1, 10), p=1),
            albu.Crop(x_min=int(box["x"]), y_min=int(box["y"]),
                      x_max=int(box["x"]+box["w"]), y_max=int(box["y"]+box["h"])),
            self._resize_correct_side(image),
            albu.PadIfNeeded(
                min_height=self.input_shape[0], min_width=self.input_shape[1], border_mode=0, value=(0, 0, 0)),
            albu.Resize(height=output_size[0], width=output_size[1]),
            # Add as many transformations as needed
            albu.Rotate(p=0.2, border_mode=cv2.BORDER_CONSTANT,),
            albu.HorizontalFlip(p=0.2),
            albu.VerticalFlip(p=0.2),
        ])

        transform_data = transform(image=image)
        new_img = transform_data["image"]
        return new_img.astype("uint8")

    def _preprocess_img_and_boxes(self, image, boxes):
        image, boxes = self._apply_albumentations(image, boxes)
        return image, boxes
