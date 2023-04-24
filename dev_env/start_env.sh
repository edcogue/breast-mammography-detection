#!/bin/bash

docker run --gpus all -p 8888:8888 -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/breast-mammography-detection:/tf/code -v /home/eduard/Documents/Master\ ciencia\ de\ dades/TFM/src/datasets:/tf/data tensorflow-full