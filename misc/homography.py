import cv2
import numpy as np

def assign_squares(img):
    height, width, _ = img.shape
    square_size = height // 8

    for i in reversed(range(8)):
        for j in range(8):
            rank = chr(ord("a") + i)
            row = 8 - j
            square = f"{rank}{row}"

            # calculate square coordinates
            x_min = (i * square_size)
            y_min = (j * square_size)
            x_max = (i * square_size) + square_size
            y_max = (j * square_size) + square_size
            x_cen = (x_min + x_max)//2
            y_cen = (y_min + y_max)//2
            
            # put text on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (0, 0, 255)
            thickness = 1
            cv2.putText(img, f"{square}", (x_cen-10, y_cen+10) , font, fontScale, color, thickness, cv2.LINE_AA)

    return img


def apply_homography(img, pts_1):
    height, width, _ = img.shape
    img_size = (height+width)//2
    pts_2 = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])

    # calculate matrix based on mapping pts_1 to pts_2
    M = cv2.getPerspectiveTransform(pts_1, pts_2)

    # apply homography
    dst = cv2.warpPerspective(img, M, (img_size, img_size))

    return dst


def apply_homography_2(img, pts_1):
    height, width, _ = img.shape
    img_size = (height+width)//2
    pts_2 = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])

    # calculate matrix based on mapping pts_1 to pts_2
    M = cv2.getPerspectiveTransform(pts_1, pts_2)
    tx, ty = img_size//6, img_size//6
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)


    print(np.int32(T) @ np.int32(M))

    # combine homography with translation
    M = T @ M

    # apply homography
    img_size = img_size+((img_size//6)*2)
    dst = cv2.warpPerspective(img, M, (img_size, img_size))

    return dst


def apply_homography_3(img, pts_1):
    height, width, _ = img.shape
    img_size = (height+width)//2
    pts_2 = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])

    # calculate matrix based on mapping pts_1 to pts_2
    M = cv2.getPerspectiveTransform(pts_1, pts_2)
    tx, ty = (img_size//2)*3, (img_size//2)*3
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)


    print(np.int32(T) @ np.int32(M))

    # combine homography with translation
    M = T @ M

    # apply homography
    img_size = img_size+((img_size//2)*6)
    dst = cv2.warpPerspective(img, M, (img_size, img_size))

    return dst