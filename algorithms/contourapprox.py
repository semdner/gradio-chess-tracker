import cv2
import numpy as np


def contour_approx(path_to_image):
    # read image
    img = cv2.imread(path_to_image)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply bilateral filter to reduce
    # filtered = cv2.bilateralFilter(gray, 15, 10, 10)
    filtered = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)

    # calculate lower and upper bound for canny edge detection
    v = np.median(img)
    sigma = 0.4
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # apply canny edge detection
    canny = cv2.Canny(filtered, lower, upper)

    # draw contours and extract the largest one
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea)

    # reduce contour to points using contour approximation
    epsilon = 0.1 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)
    corners = corners.reshape(-1, corners.shape[-1])

    return corners