import numpy as np
import cv2



def getBBOfPlyObject(image):
    proj_2d_p = np.loadtxt('corners.txt')
    proj_2d_p = proj_2d_p.astype(int).T

    # Make empty black image
    height, width, channels = image.shape

    color_pr = (0, 255, 255)
    image[proj_2d_p[1, :], proj_2d_p[0, :]] = color_pr

    # Draw the front of the bounding box
    pts = np.array(
        [[proj_2d_p[0, 1], proj_2d_p[1, 1]], [proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 4], proj_2d_p[1, 4]],
         [proj_2d_p[0, 2], proj_2d_p[1, 2]]], np.int32)
    cv2.polylines(image, [pts], True, color_pr)

    # Draw lower base of 3d bb
    pts = np.array(
        [[proj_2d_p[0, 1], proj_2d_p[1, 1]], [proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 7], proj_2d_p[1, 7]],
         [proj_2d_p[0, 5], proj_2d_p[1, 5]]], np.int32)
    cv2.polylines(image, [pts], True, color_pr)

    # Draw upper base of 3d bb
    pts = np.array(
        [[proj_2d_p[0, 6], proj_2d_p[1, 6]], [proj_2d_p[0, 2], proj_2d_p[1, 2]], [proj_2d_p[0, 4], proj_2d_p[1, 4]],
         [proj_2d_p[0, 8], proj_2d_p[1, 8]]], np.int32)
    cv2.polylines(image, [pts], True, color_pr)

    return image


