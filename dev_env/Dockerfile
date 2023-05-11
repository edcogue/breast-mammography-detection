FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-contrib-python pandas n2v albumentations pydicom beautifulsoup4 lxml
RUN pip install tensorflow_addons
RUN pip install -q datasets transformers evaluate timm pillow
RUN pip install ultralytics hyperopt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install optuna
RUN pip install --upgrade accelerate
RUN pip install pycocotools
