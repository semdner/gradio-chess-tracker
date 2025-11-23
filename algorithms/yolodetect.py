import cv2
import numpy as np
import torch
from ultralytics import YOLO


def yolo_detect(path_to_image):
    model = YOLO("model/chessboard_corners.pt")

    img = cv2.imread(path_to_image)

    results = model.predict(img, conf = 0.5, verbose = False)

    corners = []

    for pred in results:
        boxes = pred.boxes.xywh
        for box in boxes:
            x, y = box[0], box[1]
            corners.append([x, y])

    corners = np.array(corners, dtype=np.float32).reshape(-1, 2)
    return corners