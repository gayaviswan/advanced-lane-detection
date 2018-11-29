import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from globalvars import *


class Threshold:

    def __init__(self):
   
        self.original_image = []
        self.thresholded_image = []
        

    def abs_sobel_thresh(self, gray, orient='x', thresh = (0,255)):
        """
        Apply Sobel Threshold to detect edges
        
        """
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobel = np.absolute(sobel)
        binary_output = np.uint8(255 * abs_sobel/np.max(abs_sobel))
        threshold_mask = np.zeros_like(binary_output)
        threshold_mask[(binary_output >= thresh[0]) & (binary_output <= thresh[1])] = 1
        return threshold_mask

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        directional of the gradient is arctan of u gradient over x gradient
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        direction = np.absolute(direction)
        mask = np.zeros_like(direction)
        mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return mask

    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        """
        Magnitude of the gradient (square root of the squares of x and y gradients

        """
        # Calculate gradient magnitude
        # Apply threshold
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobelx2 = sobelx ** 2
        sobely2 = sobely ** 2
        abs_sobelxy =  np.sqrt(sobelx2+ sobely2)
        scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        return mag_binary

    def apply_threshold(self, img):

        """
        Combining various color, sobel, directional and magnitude of the gradient
        """
    
        self.original_image = img
        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        height, width = gray.shape
    
        # apply gradient threshold on the x axis
        sx_binary = self.abs_sobel_thresh(gray, 'x', thresh = SX_BINARY)
    
        mag_binary = self.mag_thresh(gray, sobel_kernel=KERNEL_SIZE, mag_thresh=MAG_BINARY)
        dir_binary = self.dir_threshold(gray, thresh=DIR_BINARY)
    
        grad_cond = ((sx_binary == 1) & (dir_binary == 1))
    
        color_threshold = COLOR_THRESHOLD
        r_channel = img[:,:,0]
        g_channel = img[:,:,1]
        color_combined = np.zeros_like(r_channel)
        #r_g_combo = (r_channel > color_threshold) & (g_channel > color_threshold)
        r_combo = (r_channel > color_threshold[0]) & (r_channel <= color_threshold[1])
        g_combo = (g_channel > color_threshold[0]) & (g_channel <= color_threshold[1])

        # color channel thresholds
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        l_channel = hls[:,:,1]
    
        s_thresh = S_THRESH
        s_combo = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])
    
        l_thresh = L_THRESH
        l_combo = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])

        # combine all the thresholds
        # A pixel should either be a yellowish or whiteish
        # And it should also have a gradient, as per our thresholds
        #color_combined[(r_g_combo & l_combo) & (s_combo | combined_condition)] = 1
        color_combined[(r_combo & g_combo & l_combo) & (s_combo | grad_cond)] = 1
    
        # apply the region of interest mask
        mask = np.zeros_like(color_combined)
        region_of_interest = np.array([[0,height-1], [width/2, int(0.5*height)],\
                                      [width-1, height-1]], dtype=np.int32)
        cv2.fillPoly(mask, [region_of_interest], 1)
        self.thresholded_image = cv2.bitwise_and(color_combined, mask)
    
        return self.thresholded_image
    
    
    
