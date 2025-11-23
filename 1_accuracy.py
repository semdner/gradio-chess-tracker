import os
import json
import numpy as np
from dotenv import load_dotenv, find_dotenv
from misc.config import PATH_TO_IMAGES, IMAGES, PATH_TO_VALUES, PATH_TO_GT
from misc.helper import is_board_detected, get_ground_truth
from algorithms.contourapprox import contour_approx
from algorithms.yolodetect import yolo_detect
from algorithms.yolosegment import yolo_segment
from algorithms.cameracalibration import camera_calibration
from algorithms.harriscorner import harris_corner
from math import floor

def main(algorithm):   
    total = 0
    successfull = 0

    gt_data = get_ground_truth()
    corners = None

    gt_lookup = {
        image_json["file_name"]: image_json for image_json in gt_data["data"]
    }

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

            if corners is not None and len(corners) == 4 and is_board_detected(gt_corners, corners):
                successfull += 1

            total += 1

    print(f"{algorithm} {successfull} of {total} correctly identified, percentage: {round(successfull/total, 2)}")


if __name__ == "__main__":
    load_dotenv()
    ALGORITHM = os.environ.get("ALGORITHM")

    if ALGORITHM is not None:
        ALGORITHM = str.lower(ALGORITHM)
        main(ALGORITHM)