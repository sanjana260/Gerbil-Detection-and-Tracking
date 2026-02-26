import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from ultralytics import YOLO

data_path = "data/Gerbil tracking test.v3i.yolov8/data.yaml"
train_path = "data/Gerbil tracking test.v3i.yolov8/train/"
test_path = "data/Gerbil tracking test.v3i.yolov8/test/"
val_path = "data/Gerbil tracking test.v3i.yolov8/val/"


##### I have defined the model as YOLOv8 large, with image size 832

model = YOLO("yolov8l-seg.pt")

model.train(
    data=data_path,
    epochs=60,
    imgsz=832,
    batch=8,
    device='mps',
    patience=15,
    workers=2,
    project="gerbil_training",
    name="yolov8s_seg_finetune_768",
    pretrained=True
)