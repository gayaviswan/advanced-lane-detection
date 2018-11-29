import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from calibrate import Calibrate
from threshold import Threshold
from transform import Transform
from linedetect import LineDetect
from line import Line
from globalvars import *

leftLines = Line()
rightLines = Line()

c = None

def average_line(prev_lines, new_line):
   
    num_frames = NUM_FRAMES
    
    if new_line is None:
        if len(prev_lines) == 0:
            return prev_lines, new_line
        else:
            return prev_lines, prev_lines[-1]
    else:
        if len(prev_lines) < num_frames:
            prev_lines.append(new_line)
            return prev_lines, new_line
        else:
            prev_lines[0:num_frames-1] = prev_lines[1:]
            prev_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += prev_lines[i]
            new_line /= num_frames
            return prev_lines, new_line
        
def process_image(image):

    global c
    undistort_road = cv2.undistort(image, c.mtx, c.dist, None, \
                     c.mtx)

    image_shape = image.shape

    thresh = Threshold()
    result = thresh.apply_threshold(undistort_road)

    trans = Transform()
    warped, M = trans.unwarp(result)

    linedet = LineDetect()

    ignore_frame = False

    if leftLines.detected == False or rightLines.detected == False or leftLines.fitx == None \
        or rightLines.fitx == None:
        left_fitx, right_fitx, ploty, left_fit, right_fit = linedet.fit_polynomial(warped)
    else:
        left_fitx, right_fitx, ploty = linedet.search_around_poly(warped)
    
    if left_fitx is None or right_fitx is None:
        left_fitx, right_fitx, ploty, left_fix, right_fix = linedet.fit_polynomial(warped)    

    if ignore_frame:
        if len(leftLines.past_good_fit) == 0 and len(rightLines.past_good_fit) == 0:
            return image
        else:
            left_fitx = leftLines.past_good_fit[-1]
            right_fitx = rightLines.past_good_fit[-1]
    else:
        if leftLines.past_good_fit != None:
            leftLines.past_good_fit, left_fitx = average_line(leftLines.past_good_fit, \
                                                 left_fitx)
        if rightLines.past_good_fit != None:
            rightLines.past_good_fit, right_fitx = average_line(rightLines.past_good_fit, \
                                                   right_fitx)

    if left_fitx is not None and right_fitx is not None and ploty is not None:  
        left_curverad, right_curverad = linedet.measure_curvature_real(left_fitx, right_fitx, ploty)
        offset = linedet.offset_center(left_fitx, right_fitx)  
    
        leftLines.radius_of_curvature = left_curverad 
        rightLines.radius_of_curvature = right_curverad
    else:
        if leftLines.radius_of_curvature is not None:
             left_curverad = leftLines.radius_of_curvature
        if rightLines.radius_of_curvature is not None:
             right_curverad = rightLines.radius_of_curvature
 
    if left_fitx is None or right_fitx is None or ploty is None:
        return image
    
    if left_fitx.shape != ploty.shape or right_fitx.shape != ploty.shape:
        return image
    
    fin_image = trans.inverse_warp(warped, M, undistort_road, left_fitx, right_fitx, ploty)
    
    avg_curvature = (left_curverad + right_curverad)//2 
    rad_str = "Average Radius : %.2f m " %  avg_curvature
    off_str = "Center Offset : %.2f m " % offset

    font =  cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(fin_image, rad_str, (60,60), font, 1.2, (0,255,0), 2, \
                cv2.LINE_AA)
    cv2.putText(fin_image, off_str, (100,100), font, 1.2, (0,255,0), 2,\
                cv2.LINE_AA)
    
    
    return fin_image
    

def main():
    global c
    c = Calibrate()
    o, i = c.find_chessboard_corner("./camera_cal/calibration*.jpg")
    first_image = mpimg.imread('./camera_cal/calibration1.jpg')
    image_shape = first_image.shape
    gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    mtx, dist = c.calibrate_camera(gray.shape[:2])

    undistorted = cv2.undistort(first_image, mtx, dist, None, mtx)

    """
    img = mpimg.imread('./test_images/test1.jpg')
    processed = process_image(img)

    # Plot the 2 images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(processed, cmap='gray')
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    """

    white_output = './output_images/project_video.mp4'

    clip1 = VideoFileClip('./project_video.mp4')
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


    white_output = './output_images/challenge_video.mp4'

    clip1 = VideoFileClip('./challenge_video.mp4')
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


if __name__ == "__main__":
     main()


