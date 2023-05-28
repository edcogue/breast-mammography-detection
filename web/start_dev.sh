docker run -p 8080:8080 -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection/web:/app \
-v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/outputs/models:/app/models/denoiser \
-v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection/vit-object-detection/:/app/models/vit_object_detection \
-v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection/vit-classification/:/app/models/vit_classifier \
-v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection/YOLOv8/object_detector/runs/detect/train2/weights:/app/models/yolo_object_detector \
-v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection/YOLOv8/classificator/runs/classify/train38/weights:/app/models/yolo_classifier \
mammographies_web

docker rm $(docker stop $(docker ps -a -q --filter ancestor=mammographies_web --format="{{.ID}}"))