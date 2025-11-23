import cv2
import numpy as np
import matplotlib

def harris_corner(path_to_image):
    img = cv2.imread(path_to_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)

    # new_img = np.float32(gray)
    # dst = cv2.cornerHarris(new_img, 8, 3, 0.04)
    # dst = cv2.dilate(dst, None)
        
    # threshold = 0.01 * dst.max()
    # corners = np.argwhere(dst > threshold)

    new_img = np.float32(gray)
    dst = cv2.cornerHarris(new_img, 8, 3, 0.04)
    dst = cv2.dilate(dst, None)
        
    threshold = 0.01 * dst.max()
    harris_mask = dst > threshold
    pts_y, pts_x = np.where(harris_mask)
    corners = np.vstack((pts_x, pts_y)).T

    sum_coordinates = corners.sum(axis=1)
    # for all pair [x, y] in the array calculate the difference (x - y) => return array in format [[diff_0], [diff_1], ..., [diff_n]]
    diff_coordinates = np.diff(corners, axis=1)
    # change array from [[diff_0], [diff_1], ..., [diff_n]] to [diff_0, diff_1, ..., diff_n]
    diff_coordinates = diff_coordinates.reshape(-1)

    top_left = corners[np.argmin(sum_coordinates)]      # np.argmin returns index of smallest value in arrays
    top_right = corners[np.argmin(diff_coordinates)]
    bottom_left = corners[np.argmax(diff_coordinates)]  # np.argmax returns index of biggest  value in arrays
    bottom_right = corners[np.argmax(sum_coordinates)]

    corners = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.float32).reshape(-1, 2)
    return corners
