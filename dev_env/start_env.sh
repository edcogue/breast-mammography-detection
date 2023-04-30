#!/bin/bash

docker run --gpus all -p 8888:8888 -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection:/tf/code -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/datasets:/tf/data -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/datasets/Mammographies:/kaggle/input/mias-cbis-ddsm-inbreast/Mammographies -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/outputs:/kaggle/working tensorflow-full
