#!/bin/sh
# open CPU models used by the AV-QA visual pass (avqa_faces.py), downloaded into ./models
set -e
mkdir -p models
curl -sL -o models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# python deps: pip install opencv-python-headless mediapipe numpy matplotlib (mediapipe also needs libEGL1 + libGLES2 on Linux)
