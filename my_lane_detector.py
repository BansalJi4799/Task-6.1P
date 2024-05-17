#!/usr/bin/env python3

#Python Libs
import sys, time

#numpy
import numpy as np

#OpenCV
import cv2
from cv_bridge import CvBridge

#ROS Libraries
import rospy
import roslib

#ROS Message Types
from sensor_msgs.msg import CompressedImage

class Lane_Detector:
    def __init__(self):
        self.cv_bridge = CvBridge()

        #### REMEMBER TO CHANGE THE TOPIC NAME! #####        
        self.image_sub = rospy.Subscriber('/oscarducky/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)
        #############################################

        rospy.init_node("my_lane_detector")

    def image_callback(self,msg):
        rospy.loginfo("image_callback")


        # Convert to opencv image 
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        
        #### YOUR CODE GOES HERE ####

        # flip along the horizontal axis using an OpenCV function
        img_out = cv2.flip(img, 0)

        print("Image is of type: ", type(img))
        print("No. of dimensions: ", img.ndim)
        print("Shape of image: ", img.shape)
        print("Size of image: ", img.size)
        print("Image stores elements of type: ", img.dtype)


        img_out = img[150:470, 1:630]
        print("Shape ", img_out.shape)

        img_out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # White Line
        #lower_blue = np.array([200,200,200])
        #upper_blue = np.array([255,255,255])
        #mask_blue = cv2.inRange(img, lower_blue, upper_blue)
        #img_out = cv2.bitwise_and(img, img, mask = mask_blue)
 
        lower_blue = np.array([10,100,100])
        upper_blue = np.array([100,255,255])
        mask_blue = cv2.inRange(img, lower_blue, upper_blue)
        img_out = cv2.bitwise_and(img, img, mask = mask_blue)
        
        grey = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
        grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
        
        #img_out = cv2.cvtColor(img, cv2.COLOR_)
        # Defining all the parameters 
        t_lower = 100 # Lower Threshold 
        t_upper = 255 # Upper threshold 
        aperture_size = 7 # Aperture size       
        L2Gradient = True # Boolean 
  
        # Applying the Canny Edge filter  
        # with Aperture Size and L2Gradient 
        
        edge = cv2.Canny(img_out, t_lower, t_upper, 
                 apertureSize = aperture_size,  
                 L2gradient = L2Gradient )

        #############################

        rho = 1
        theta = np.pi /180
        threshold = 180
        min_line_length = 50
        max_line_gap = 10
        
        lines = cv2.HoughLinesP(grey, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        
       
        # Show image in a window
        cv2.imshow('img_out', self.output_lines(img, lines))
        cv2.waitKey(1)

   
    def output_lines(self, original_image, lines):
        output = np.copy(original_image)
        if lines is not None:
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0],l[1]), (l[2],l[3]), (0,0,255), 2, cv2.LINE_AA)
                #cv2.circle(output, (l[0],l[1]), 2, (0,255,0))
                #cv2.circle(output, (l[2],l[3]), 2, (0,0,255))
        return output

    def run(self):
    	rospy.spin() # Spin forever but listen to message callbacks

if __name__ == "__main__":
    try:
        lane_detector_instance = Lane_Detector()
        lane_detector_instance.run()
    except rospy.ROSInterruptException:
        pass
    
    
