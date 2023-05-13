#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int16
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from math import dist, atan2
from keras.models import load_model

class ImgClass(Node):

    def __init__(self):		
        
        print("Starting Img Classification Node...")

        # Creates the node.
        super().__init__("img_class")

        self.declare_parameter('modelDir', '/home/himanshu/hr_7785/src/hr13_lab6/hr13_lab6/')

        self.model = load_model( self.get_parameter('modelDir').value + "new_model.h5")

        self.size = (30, 30)

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1
        )

        self.proc_img_subs = self.create_subscription(CompressedImage, "processed_img", self.callback_proc_img, image_qos_profile)

        self.class_pub = self.create_publisher(Int16, "img_class", 10)


    def callback_proc_img(self, msg):

        proc_img = CvBridge().compressed_imgmsg_to_cv2(msg, "bgr8")
        test_img = np.array([np.array(cv2.resize(proc_img, self.size))])
        pred_clasess = self.model.predict(test_img)
        pred_labels = np.argmax (pred_clasess, axis = 1)
        print("Image Class: " , pred_labels)
        msg = Int16()
        msg.data = int(pred_labels)
        self.class_pub.publish(msg)

    
def main():
    rclpy.init() #init routine needed for ROS2.
    img_class_node = ImgClass()

    rclpy.spin(img_class_node)

    img_class_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
