FROM --platform=linux/amd64 python:3.9-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir -p image_uploads image_results
RUN pip3 install albumentations
RUN pip3 install n2v
RUN pip3 install tensorflow-cpu
RUN pip3 install transformers timm pillow
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install ultralytics
#COPY . .
#CMD [ "python3", "app.py"]
CMD ["flask", "run",  "--host=0.0.0.0","--port=8080", "--debug"]