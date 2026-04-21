import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from ultralytics import YOLO
from arena_annotation import *

# %matplotlib notebook
matplotlib.use("module://matplotlib_inline.backend_inline")

import cv2
import numpy as np
import sys

CONF_THRESH = 0.5
IOU_THRESH = 0.4
MIN_CLASS_CONF = 0.4  # minimum confidence to accept a class detection
X_DIST = 200
Y_DIST = 150

if len(sys.argv)<4:
    raise AttributeError("Please Enter the correct Arguments: Video input folder, Video name string, Desired video output folder, and CSV output folder.")
VIDEO_INPUT_FOLDER = sys.argv[1]
VIDEO_NAME_STRING  = sys.argv[2]
VIDEO_OUTPUT_FOLDER = sys.argv[3]
CSV_OUTPUT_FOLDER = sys.argv[4]

initial_points = np.load(
    "Arena_base.npy"
)

video_list = glob.glob(f"{VIDEO_INPUT_FOLDER}/*{VIDEO_NAME_STRING}*.mp4")
if len(video_list)==0:
    raise ValueError("No Videos in requested format")

cap = cv2.VideoCapture(video_list[0])

ret, first_frame = cap.read()

updated_points = adjust_rectangle(
    first_frame,
    initial_points
)

print(updated_points)

def mask_centroid_fast(mask):
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    h, w = mask.shape

    ys, xs = torch.nonzero(mask, as_tuple=True)

    if xs.numel() == 0:
        return None

    cx = xs.float().mean()
    cy = ys.float().mean()

    return cx.item(), cy.item()

import torch

def mask_centroid(mask):
    # ensure shape [H, W]
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    # get coordinates grid
    h, w = mask.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=mask.device),
        torch.arange(w, device=mask.device),
        indexing="ij"
    )

    # flatten
    mask_flat = mask.reshape(-1)
    x_flat = x_coords.reshape(-1)
    y_flat = y_coords.reshape(-1)

    # avoid empty masks
    total = mask_flat.sum()
    if total == 0:
        return None  # or (nan, nan)

    cx = (x_flat * mask_flat).sum() / total
    cy = (y_flat * mask_flat).sum() / total

    return cx.item(), cy.item()


def get_video_output(VIDEO_INPUT, VIDEO_OUTPUT, CSV_OUTPUT, frame_coords):
    print("Started Video Processing")
    x_0 = (frame_coords[0][0] + frame_coords[3][0])/2
    x_max = (frame_coords[1][0] + frame_coords[2][0])/2
    y_0 = (frame_coords[0][1] + frame_coords[1][1])/2
    y_max = (frame_coords[2][1] + frame_coords[3][1])/2

    cap = cv2.VideoCapture(VIDEO_INPUT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    import numpy as np
    import csv

    # YOLOv8s-seg.pt with 80 epochs
    yolov8s_80epochs_640 = "runs/segment/gerbil_training/yolov8s_seg_finetune_640/weights/best.pt"

    model = YOLO(yolov8s_80epochs_640)

    # Fixed BGR colors per class (OpenCV uses BGR)
    CLASS_COLORS = {
        0: (255, 0, 0),     # Blue
        1: (0, 255, 0),     # Green
        2: (0, 0, 255),     # Red
        3: (255, 255, 0),   # Cyan
        4: (255, 0, 255),   # Magenta
    }

    # -----------------------------
    # Tracking data store
    # {frame_idx: [{"class_id": int, "class_name": str, "center_x": int, "center_y": int, "conf": float}]}
    # -----------------------------
    tracking_data = []
    frame_idx = 0

    # -----------------------------
    # Tracking loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            agnostic_nms=True,
            persist=True,
            verbose=False
        )

        result = results[0]
        # tracking_data[frame_idx] = []  # initialise entry for this frame

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes
            masks = result.masks

            cls = boxes.cls
            conf = boxes.conf
            xyxy = boxes.xyxy

            keep_indices = []

            # -----------------------------
            # Enforce ONE detection per class
            # -----------------------------
            for c in torch.unique(cls):
                class_indices = torch.where(cls == c)[0]
                class_confs = conf[class_indices]

                best_idx_local = torch.argmax(class_confs)
                best_idx = class_indices[best_idx_local]

                if class_confs[best_idx_local] > MIN_CLASS_CONF:
                    keep_indices.append(best_idx)

            if len(keep_indices) > 0:
                keep_indices = torch.stack(keep_indices)

                filtered_boxes = xyxy[keep_indices]
                filtered_cls = cls[keep_indices]
                filtered_conf = conf[keep_indices]
                filtered_masks = masks.data[keep_indices] if masks is not None else None

                # -----------------------------
                # Draw results
                # -----------------------------
                for i in range(len(filtered_boxes)):
                    x1, y1, x2, y2 = filtered_boxes[i].int().tolist()
                    class_id = int(filtered_cls[i].item())
                    confidence = float(filtered_conf[i].item())
                    mask = filtered_masks[i]

                    center_x, center_y = mask_centroid(mask)
                    # print(center_x, center_y)

                    center_x = int(center_x)
                    center_y = int(center_y)

                    # -----------------------------
                    # Calculate and store centre
                    # -----------------------------
                    # center_x = (x1 + x2) // 2
                    # center_y = (y1 + y2) // 2

                    x = round((center_x - x_0)/(x_max - x_0) * X_DIST,2)
                    y = round((center_y - y_0)/(y_max - y_0) * Y_DIST,2)

                    tracking_data.append({
                        "frame_id":   frame_idx,
                        # "class_id":   class_id,
                        # "class_name": model.names[class_id],
                        "center_x":   x,
                        "center_y":   y,
                        # "conf":       round(confidence, 4),
                    })

                    label = f"{model.names[class_id]} {confidence:.2f}"
                    # color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    color = (255, 255, 255)

                    # # Draw bounding box
                    cv2.circle(frame, (center_x, center_y), radius=5, color=color, thickness=-1)

                    label = f"({x},{y})"

                    cv2.putText(
                        frame,
                        label,
                        (center_x + 10, center_y - 10),  # offset so text doesn't overlap the circle
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,          # font scale
                        (255, 255, 255),  # text color (BGR)
                        1,            # thickness
                        cv2.LINE_AA
                    )

                    # Draw mask if segmentation
                    if filtered_masks is not None:
                        mask = filtered_masks[i].cpu().numpy()

                        # Resize mask to frame size
                        mask = cv2.resize(mask, (width, height))

                        # Create boolean mask
                        mask_bool = mask > 0.5

                        # Define overlay color (BGR)
                        overlay_color = np.array(list(color), dtype=np.float32)

                        # Convert frame to float for blending
                        frame_float = frame.astype(np.float32)

                        # Blend only where mask is true
                        # frame_float[mask_bool] = (
                        #     frame_float[mask_bool] * 0.5 + overlay_color * 0.5
                        # )

                        frame = frame_float.astype(np.uint8)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # -----------------------------
    # Save tracking data to CSV
    # -----------------------------

    tracking_df = pd.DataFrame(tracking_data)
    tracking_df.to_csv(CSV_OUTPUT,index=False)

    print("Finished. Saved video to:", VIDEO_OUTPUT)
    print("Saved centre coordinates to:", CSV_OUTPUT)

print(f"Processing {len(video_list)} Videos")
for video in video_list:
    print(f"Processing {video}")
    VIDEO_INPUT = video
    video_name = os.path.splitext(os.path.basename(video))[0]
    VIDEO_OUTPUT = f"{VIDEO_OUTPUT_FOLDER}/{video_name}.mp4"
    CSV_OUTPUT = f"{CSV_OUTPUT_FOLDER}/{video_name}.csv"
    get_video_output(VIDEO_INPUT,VIDEO_OUTPUT,CSV_OUTPUT, updated_points)