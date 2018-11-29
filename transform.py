import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from globalvars import *

class Transform:

    """
    Perspective transform of the image
    """
    def __init__(self):
        self.image = []

    def unwarp(self, image):
        """
        Perspective transform
        """
        self.image = image
        img_size = (image.shape[1], image.shape[0])
        src = np.float32(
                 [[(img_size[0] / 2) - 70, img_size[1] / 2 + 110],
                 [((img_size[0] / 6) + 10), img_size[1]],
                 [(img_size[0] * 5 / 6) + 60, img_size[1]],
                 [(img_size[0] / 2 + 43), img_size[1] / 2 + 110]])
        #src = np.float32([[570, 470], [220, 720], [1110, 720], [722,470]])

        dst = np.float32([[(img_size[0] / 4) - 40, 0],
              [(img_size[0] / 4) - 40, img_size[1]],
              [(img_size[0] * 3 / 4) + 40, img_size[1]],
              [(img_size[0] * 3 / 4) + 40, 0]])

        #dst = np.float32([[280, 0], [280, 720], [1000, 720], [1000,0]])
        # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        self.warped = cv2.warpPerspective(image, self.M, img_size)

        # Return the resulting image and matrix
        return self.warped, self.M


    def inverse_warp(self, image, M, undistort_img, left_fitx, right_fitx, ploty):
        """
        Transform the image back to original after drawing the lane line
        """

        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        ret, IM = cv2.invert(M)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, IM, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        restored = cv2.addWeighted(undistort_img, 1, newwarp, 0.3, 0)

        return restored

