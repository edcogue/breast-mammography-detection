#!/bin/bash

docker run --rm --gpus all -p 8888:8888 -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection:/workspace/code -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/datasets:/workspace/data -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/datasets/Mammographies:/workspace/input/mias-cbis-ddsm-inbreast/Mammographies -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/outputs:/workspace/working pytorch-jupyter
