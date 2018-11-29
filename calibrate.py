import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from threshold import Threshold
from transform import Transform
from globalvars import *


class Calibrate:

    """
    Calibrate and correct distortion in the images

    """
    def __init__(self):
        self.objpoints = []
        self.imgpoints = []   
        self.list_of_images = []
        self.mtx = []
        self.dist = []

    def find_chessboard_corner(self, file_path):
        """
        Using set of chessboard images to help us the calibrate our images
        of the road

        Input:
        file_name: Accepts the path to a list of file to parse 

        Output:
        objpoints: List of corners in the chessboard
        imgpoints: Corresponding image points

        """
        self.list_of_images = glob.glob(file_path)

        #prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((CHESS_BOARD_Y*CHESS_BOARD_X,3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESS_BOARD_X, 0:CHESS_BOARD_Y].T.reshape(-1,2)  

        # Step through the list and search for chessboard corners
        for fname in self.list_of_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (CHESS_BOARD_X, CHESS_BOARD_Y),None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (CHESS_BOARD_X, CHESS_BOARD_Y), corners, \
                      ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()
        return self.objpoints, self.imgpoints

    def get_points(self):

        return self.objpoints, self.imgpoints

    def calibrate_camera(self, shape):
         """
         Calibrate the camera using the objpoints and imagepoints detected using chessboard 
         images

         """
         ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, \
               self.imgpoints, shape, None, None)
         return self.mtx, self.dist

def main():
    c= Calibrate()
    o, i = c.find_chessboard_corner("./camera_cal/calibration*.jpg")
    first_image = mpimg.imread('./camera_cal/calibration1.jpg')
    image_shape = first_image.shape
    gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    mtx, dist = c.calibrate_camera(gray.shape[:2])


    undistorted = cv2.undistort(first_image, mtx, dist, None, mtx)

    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(first_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    """
    my_image = mpimg.imread('test_images/straight_lines1.jpg')
    undistort_road = cv2.undistort(my_image, mtx, dist, None, mtx)
    # Plot the 2 images side by side
    f, ax1 = plt.subplots(1, 1)
    #f.tight_layout()
    #ax1.imshow(undistort_road)
    #ax1.set_title('Original Image', fontsize=50)
    #ax2.imshow(warped, cmap='gray')
    #ax2.set_title('Thresholded Image', fontsize=50)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.savefig('./output_images/undistort_road.jpg')

  
    image_shape = my_image.shape
    
    thresh = Threshold()
    result = thresh.apply_threshold(undistort_road)
    f, ax1 = plt.subplots(1, 1)
    ax1.imshow(result, cmap="gray")
    plt.savefig('./output_images/threshold.jpg')

    trans = Transform()
    warped, M = trans.unwarp(result)
    #f, ax1 = plt.subplots(1, 1)
    #ax1.imshow(warped, cmap="gray")
    #plt.savefig('./output_images/warped.jpg')

    
if __name__ == "__main__":
     main()



