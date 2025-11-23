import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

def camera_calibration(path_to_image, start_pos):
    if start_pos:
        chessboard = (3, 7)
        img = cv2.imread(path_to_image)
        img = cv2.resize(img, None, fx=0.25, fy=0.25)
    else:
        chessboard = (7, 7)
        img = cv2.imread(path_to_image)

    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    if start_pos:
        ret, corners = cv2.findChessboardCorners(gray, chessboard, cv2.CALIB_CB_FILTER_QUADS)
        if corners is not None:
            top_left = corners[6][0]
            top_right = corners[8][0]
            bottom_left = corners[12][0]
            bottom_right = corners[14][0]
        else:
            return None
    else:
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard, cv2.CALIB_CB_FILTER_QUADS)
        if corners is not None:
            top_left = corners[0][0]
            top_right = corners[6][0]
            bottom_left = corners[42][0]
            bottom_right = corners[48][0]
        else:
            return None
    

    corners = np.array([
        bottom_left,
        top_left,
        bottom_right,
        top_right,
    ], dtype=np.float32)

    height, width, _ = img.shape
    img_size = (height+width)//2

    reprojected_corners = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])

    # calculate Homographymatrix
    M = cv2.getPerspectiveTransform(corners, reprojected_corners)

    if start_pos:
        tx, ty = (img_size//2)*3, (img_size//2)*3
    else:
        tx, ty = img_size//6, img_size//6

    # define Translationmatrix
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1 ]], dtype=np.float32)

    # multiply both matrices
    M = T @ M

    # calculate corrected image size 
    if start_pos:
        img_size = img_size+((img_size//2)*6)
    else:
        img_size = img_size+((img_size//6)*2)

    # define reprojected corners
    reprojected_corners = pts_warped = np.float32([
        [0, 0],
        [0, img_size],
        [img_size, 0],
        [img_size, img_size]
    ]).reshape(-1, 1, 2)

    # calculate reverse Homographymatrix
    M_inv = np.linalg.inv(M)

    corrected_corners = cv2.perspectiveTransform(reprojected_corners, M_inv)

    corrected_corners = np.array(corrected_corners, dtype=np.float32).reshape(-1, 2)

    if start_pos:
        corrected_corners = corrected_corners / 0.25

    return corrected_corners
