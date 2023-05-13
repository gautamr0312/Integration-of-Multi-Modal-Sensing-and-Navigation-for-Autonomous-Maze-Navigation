#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, String
from sensor_msgs.msg import CompressedImage, LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data

import sys

import numpy as np
import cv2
from cv_bridge import CvBridge
from math import pi

PI = pi
TOL = 0.1
AOV = 15 * (PI / 180)
FRAME_WIDTH = 320               # Capture frame width 
FOV = 62.2

class SensorNode(Node):

    def __init__(self):		
        
        print("Starting Img Processing Node...")

        # Creates the node.
        super().__init__("sensor_node")

        # Set Parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")
        self.declare_parameter('video_source','/camera/image/compressed')       # Parameter to change the video topic 
        self.declare_parameter("min_obs_dist", 0.25)

        self.min_object_dist = 0.65
       
        #Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
        
        #Only create image frames if we are not running headless (_display_image sets this)
        if(self._display_image):
        # Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
            cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
        
        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1
        )

        #Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subs = self.create_subscription(
                CompressedImage,
                self.get_parameter('video_source').value,
                self._image_callback,
                image_qos_profile)
        self._video_subs # Prevents unused variable warning.

        self.new_img = None

        self.new_img_pub = self.create_publisher(CompressedImage, "processed_img", 10)

        self.laser_subs = self.create_subscription(LaserScan, "scan", self.callback_laser_scan, qos_profile_sensor_data)

        self.obj_detect_pub = self.create_publisher(Bool,"/object_detect",10)

        self.state_subs = self.create_subscription(String, "/state", self.callback_state, 10)

        self.curr_state = "go"

        self.ang_wrt_lid = 0.0
        self.ang_pubs = self.create_publisher(Float64, "/angle_wrt_lidar", 10)

        self.object_dist_pub = self.create_publisher(Float64, "/object_dist", 10)


    def callback_state(self, msg : String):
        self.curr_state = msg.data
    
    def callback_laser_scan(self, msg : LaserScan):
        # print("Callback Lidar...")
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        index = round( (abs(AOV) - angle_min ) / angle_inc)


        ranges = np.asarray(msg.ranges[:index] + msg.ranges[-index:])
        ranges = ranges[~np.isnan(ranges)]
        
        min_rng_val = np.min(ranges)
        print(f"The obstacle is at {min_rng_val}")

        # msg_dist = Float64()
        # msg_dist.data = float(min_rng_val)
        # self.object_dist_pub.publish(msg_dist)

        msg_bool = Bool()
        if min_rng_val < self.min_object_dist:
            print("Object Close, trigger object detection")
            msg_bool.data = True
        else:
            msg_bool.data = False


        self.obj_detect_pub.publish(msg_bool)




    def _image_callback(self, CompressedImage):	
        # The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        # print("Img callback")
        try:
            self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
            self.show_image(self._titleOriginal, self._imgBGR)
            new_img = self.process_img(self._imgBGR)            
            self.show_image("New Image", new_img)
            self.new_img_pub.publish(CvBridge().cv2_to_compressed_imgmsg(new_img))
        except:
            print("Empty image")
        
        
    def show_image(self, title, img):
        cv2.imshow(title, img)
        # cv2.setMouseCallback(self._titleOriginal, self.mouseRGB)
        # Cause a slight delay so image is displayed
        self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

    def process_img(self, image):

        img_copy      = image.copy()
        img_copy      = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
        ## Blurring
        blurred       = cv2.medianBlur(img_copy, 7)
        filter_       = cv2.bilateralFilter(blurred, 5, 75, 75)
        ##Threshing
        _,thresh = cv2.threshold(filter_,90,255,cv2.THRESH_BINARY)
        
        ## Morphological Operations
        element       = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1)) 
        dilated       = cv2.dilate(thresh, element, iterations=10)
        
        ## Bitwise And Operation
        mask          = 255-dilated
        sample        = mask*img_copy
        
        ## Contour Detection using Canny Edge Detection Algorithm
        
        edged = cv2.Canny(sample,90,200)
        contours, _ = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        ## Find the contour with the maximum area
        try:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            self.cnt_prop(cnt)
        except:
            return image

        ## Return the cropped image
        x,y,w,h = cv2.boundingRect(cnt)
        return image[y-30:y+h+20,x-30:x+w+20]

    def cnt_prop(self, cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)
        self.ang_wrt_lid =  - (( ((x + w/2) - 160) / FRAME_WIDTH ) * FOV ) * (PI / 180)
        msg = Float64()
        msg.data = self.ang_wrt_lid
        self.ang_pubs.publish(msg)

        

def main():
    rclpy.init() #init routine needed for ROS2.
    sensor_node = SensorNode()

    rclpy.spin(sensor_node)

    sensor_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

