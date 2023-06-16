# Breast Cancer Detection

## Introduction
This repository contains all the code and Dockerfiles used at my Master's Thesis **"Aplicación de nuevos algoritmos en detección y
clasificación de anomalı́as en mamografı́as"**.

Different techniques and algorithms are used in breast cancer detection, such as:

* **Noise2Void**: denoise mammographies images (processes images of 1600x1600px).
* **ViT**: performs classification for benign and malign abnormalities.
* **DETR**: used as object detection model to locate abnormalities, all labelled as Abnormality.
* **YOLOv8**: three different models are used: as an object detector, benign/malign classifier and type of abnormality classifier.
* **Conditional GAN**: a custom architecture based on ROIMammoGAN is developed in order to augment available data for type abnormality classification.

Also, a simple containerized web application is available.

Pretrained models are available in the next link [models](https://www.kaggle.com/datasets/eduardconesa/mammographies-models). Put all the folders inside web/models folder and the web is ready to go.

New models will be available with better performance soon!

## Repository Structure
This project contains:
* **data_augmentation**: jupyter notebooks and python scripts to train the Conditional GAN.
* **dev_env**: Dockerfiles and bash_scripts to deploy the development environment. Remember to set your local paths for the volumes in the bash scripts.
* **generate_data**: jupyter notebook to generate YOLOv8 data. Also available in the link below.
* **preprocess**:
    * **data_loader**: python class to load data, perform augmentations and format for the different algorithms.
    * **noise2void**: jupyter notebooks to train and report metrics about noise2void performance.
    * **preprocess_and_data_exploration.ipynb**: preprocess original data from original kaggle datasets. Preprocessed data is available in the link below.
    * **visualize_abnormalities.ipynb**: visualizes different types of abnormalities.
* **vit-classification**: jupyter notebooks and python scripts to train ViT.
* **vit-object-detection**: jupyter notebooks and python scripts to train DETR.
* **web**: a simple web app to deploy the models using flask. Dockerfiles and scripts are available for development and release version. Remember to put your models in the models folder.
* **YOLOv8**:
    * **abnormalities_type_classificator**: train the abnormalities type classificator.
    * **classificator**: train the benign/malign classificator.
    * **object_detector**: train the object detector.



## Develope and contribute
All the environment can be easily deployed on you machines by building the Dockerfiles inside the dev_env and using the bash scripts to launch the containers. Change the paths with your local paths and enjoy a full-equiped jupyter notebook server with all dependencies installed.

Data used to train the models is available in the next links:
* [Preprocessed data for ViT, DETR, Noise2Void and Conditional GAN](https://www.kaggle.com/datasets/eduardconesa/mias-cbis-ddsm-inbreast)
* [Preprocessed data for YOLOv8 training, both classification and object detection](https://www.kaggle.com/datasets/eduardconesa/mias-cbis-ddsm-inbreast-yolo)
