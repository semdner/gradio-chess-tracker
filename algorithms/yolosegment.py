import cv2
import numpy as np
import torch
from ultralytics import YOLO

def yolo_segment(path_to_image):
    model = YOLO("model/chessboard_segmentation.pt")

    img = cv2.imread(path_to_image)

    results = model.predict(img, conf = 0.9, verbose = False)

    contour = None
    corners = None

    for pred in results:
        for index, item in enumerate(pred):
            # create mask
            binary_mask = np.zeros(img.shape[:2], np.uint8)
            contour_pred = item.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            new_img = cv2.drawContours(binary_mask, [contour_pred], -1, (255, 255, 255), cv2.FILLED)

            # create contour from mask
            contour_from_mask, hierarchy = cv2.findContours(new_img, 1, 2)

            # reduce the points of the contour
            for cnt in contour_from_mask:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                corners = cv2.approxPolyDP(cnt, epsilon, True)

    
    if corners is not None:
        corners = np.array(corners, dtype=np.float32).reshape(-1, 2)

    return corners

            

