import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from globalvars import *

class LineDetect:
    """
    Detect lines using sliding window protocol
    """
    def __init__(self):
        self.left_fitx = []
        self.right_fitx = []
        self.ploty = []
        self.right_fit = []
        self.left_fit = []
        self.warped = []

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = NWINDOWS
        # Set the width of the windows +/- margin
        margin = MARGIN
        # Set minimum number of pixels found to recenter window
        minpix = MINPIX

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_xleft_low =  leftx_current - margin# Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current  - margin# Update this
            win_xright_high = rightx_current + margin  # Update this
        
            cv2.rectangle(out_img,(win_xleft_low,win_y_low), \
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low), \
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &  \
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img


    def fit_polynomial(self, binary_warped):

        self.warped = binary_warped
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        try: 
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError:
            return None, None, None, None, None
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
    
        return left_fitx, right_fitx, ploty, left_fit, right_fit

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        try: 
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            left_fitx = None
            right_fitx = None
            print("Failed to fit a line")

        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty

        return left_fitx, right_fitx, ploty

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        result = None
        margin = MARGIN_SEARCH_POLY
        self.warped = binary_warped

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +  \
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + \
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + \
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + \
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, \
                                       rightx, righty)
    
        return left_fitx, right_fitx, ploty


    def measure_curvature_real(self, left_fitx, right_fitx, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = YM_PER_PIX # meters per pixel in y dimension
        xm_per_pix = XM_PER_PIX # meters per pixel in x dimension
    
        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
        #ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
    
    
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx* xm_per_pix, 2)
    
        left_curverad =  ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*left_fit_cr[0]) 
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)\
                         / np.absolute(2*right_fit_cr[0])
    
        return left_curverad, right_curverad

    def offset_center(self, left_fitx, right_fitx):
        xm_per_pix = XM_PER_PIX
        center_lane = (right_fitx[self.warped.shape[0]-1] + left_fitx[self.warped.shape[0]-1])/2
        #center_lane = (rightx[0] + leftx[0])/2
    
        center_car = self.warped.shape[1]//2
        offset = center_lane - center_car
    
        center_offset_pixels = abs(center_car - center_lane)
        center_offset_meters = xm_per_pix*center_offset_pixels
    
        return center_offset_meters


