import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from misc.config import PATH_TO_GT

TOLERANCE = 80


"""
Opens the file with the ground truth data of all images, stores it
in a json object and returns it. 
"""
def get_ground_truth():
    try:
        with open(PATH_TO_GT, 'r') as f:
            ground_truth_data = json.load(f)
            return ground_truth_data
    except FileNotFoundError:
        exit("ERROR: The file 'data.json' was not found.")

"""
Calculate the euclidean distance for all predicted points to the ground truth points.
Returns True if all points match/are within the tolerance, False if not
"""
def is_board_detected(gt_points, pred_points):
    dists = np.linalg.norm(gt_points[:,None,:] - pred_points[None,:,:], axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    success = np.all(matched_dists <= TOLERANCE)

    return success


def euclidean_distance(gt_points, pred_points):
    dists = np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    return np.sum(matched_dists)


def pck_distance(gt_points, pred_points):
    dists = np.linalg.norm(gt_points[:,None,:] - pred_points[None,:,:], axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    within_tolerance = np.sum(matched_dists <= TOLERANCE)

    return within_tolerance