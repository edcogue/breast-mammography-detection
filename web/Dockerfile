FROM --platform=linux/amd64 python:3.9-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
RUN pip install flask
RUN pip install albumentations
RUN pip install n2v
RUN pip install tensorflow-cpu
RUN pip install transformers timm pillow
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install ultralytics
RUN pip install opencv-python
RUN pip install dill
CMD ["flask", "run",  "--host=0.0.0.0","--port=8080", "--debug"]