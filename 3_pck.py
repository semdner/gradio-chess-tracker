import os
import json
import numpy as np
from dotenv import load_dotenv, find_dotenv
from misc.config import PATH_TO_IMAGES, IMAGES, PATH_TO_VALUES, PATH_TO_GT
from misc.helper import pck_distance, get_ground_truth
from algorithms.harriscorner import harris_corner
from algorithms.contourapprox import contour_approx
from algorithms.yolodetect import yolo_detect
from algorithms.yolosegment import yolo_segment
from algorithms.cameracalibration import camera_calibration
from algorithms.harriscorner import harris_corner


def main(algorithm):
    total = 0
    within_tolerance = 0

    gt_data = get_ground_truth()
    gt_lookup = {
        image_json["file_name"]: image_json for image_json in gt_data["data"]
    }
    corners = None

    for image in IMAGES:
        if image in gt_lookup:
            path_to_image = f"{PATH_TO_IMAGES}/{image}"

            gt_corners = np.array(gt_lookup[image]["keypoints"])
            gt_corners = gt_corners.reshape(-1, 2)

            match algorithm:
                case "harris_corner_detection":
                    corners = harris_corner(path_to_image)
                case "contour_approx":
                    corners = contour_approx(path_to_image)
                case "camera_calibration":
                    corners = camera_calibration(path_to_image, False)
                    if corners is None:
                        corners = camera_calibration(path_to_image, True)
                case "line_detection":
                    print("line_detection")
                case "yolo_detect":
                    corners = yolo_detect(path_to_image)
                case "yolo_segment":
                    corners = yolo_segment(path_to_image)
                case _:
                    exit("ERROR: No valid algorithm specified")
# 
            if corners is not None:
                within_tolerance += pck_distance(pred_points=corners, gt_points=gt_corners)
            
            total += 4

    PCK = round((within_tolerance/total), 2)

    print(f"{algorithm} PCK: {PCK}, within tolerance: {within_tolerance}, total: {total}")


if __name__ == "__main__":
    load_dotenv()
    ALGORITHM = os.environ.get("ALGORITHM")

    if ALGORITHM is not None:
        ALGORITHM = str.lower(ALGORITHM)
        main(ALGORITHM)