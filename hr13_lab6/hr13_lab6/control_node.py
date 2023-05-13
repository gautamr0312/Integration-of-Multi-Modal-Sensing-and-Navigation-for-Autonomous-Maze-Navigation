#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Bool, String, Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan  
from rclpy.qos import qos_profile_sensor_data 
from turtlebot3_msgs.msg import Sound
import numpy as np
import math
import time

LEFT_ANG_SETPT = - np.pi / 2
RIGHT_ANG_SETPT = np.pi / 2
ANG_SETPT = 0.0
TOL = 0.2
DIST_SETPT = 0.90

class Controller(Node):

    def __init__(self):		
        
        print("Starting Controller Node...")

        # Creates the node.
        super().__init__("controller")


        # Subcription to the topic classification topic
        self.img_class_subs = self.create_subscription(Int16, "img_class", self.callback_img_class, 10)

        # Odom Subscriber
        self.odom_subs = self.create_subscription(Odometry, "odom", self.callback_odom, 10)

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Laser Subscriber
        self.laser_dist_subs = self.create_subscription(Bool, "/object_detect", self.callback_laser_scan, 10)
        
        # Angle wrt Lidar
        self.kp_ang = 0.42
        self.kp_ang_ah = 0.30
        self.ang_flag = False
        self.angle_wrt_lidar = 0.0
        self.ref_ang = 0.0
        self.ang_err = 0.0
        self.angle_subs = self.create_subscription(Float64,
                                                   "/angle_wrt_lidar",
                                                   self.callback_angle_lidar,
                                                   1)
        
        self.object_detect_subs = self.create_subscription(Float64,
                                                           "/object_dist",
                                                           self.callback_object_dist,
                                                           1)

        # Speeds
        self.default_rotate_speed = 0.25 # Default Rotating speed 
        self.action_rotate_speed = 0.35 # Action Rotate Speed 
        self.straight_speed = 0.15

        # Flag use to reset the position values
        self.Init = True

        # Initial Values of Pose
        self.Init_posx = 0.0
        self.Init_posy = 0.0
        self.Init_posz = 0.0

        # Current position and orientation of the robot in the global 
        # reference frame
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.last_theta = 0.0
        self.ang_setpoint = 0.0

        # Angular Controller
        self.ang_tolerance =  1e-1
        self.kp_angular = 0.75

        self.sym_detection = True

        self.img_class = 0
        self.current_class = None
        
        # States of the state-machine
        # go : go straight
        # detection : detect class from ML model
        # action : take action corresponding to class
        # adjust_heading : recovery mode
        self.state = "go"

        # State Publisher 
        self.state_pub = self.create_publisher(String, "/state", 10)

        self.state_timer = self.create_timer(0.3, self.state_machine)

        self.sum = 0 

        self.object_dist = 0.0
        self.kp_lin = 1.0
        
        self.stop_flag = False

        # self.prev_class = 0
        self.turn_dir = 1

        #Laser Tolerance
        self.laser_angles = np.array([0.0, 30.0, 90.0, 270.0, 330.0]) * (np.pi / 180 )
        self.laser_tolerance = 5

        # Lidar Subsriber
        self.lidar_subs = self.create_subscription(LaserScan,
                                                   "/scan",
                                                   self.callback_scan,
                                                   qos_profile_sensor_data)
        
    def callback_scan(self, msg):

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        len_rng_arr = len(msg.ranges)
        distance = []
        for angle in self.laser_angles:

            index = round( (abs(angle) - angle_min ) / angle_inc) 
            index_min = round(index - self.laser_tolerance)
            index_max = round(index + self.laser_tolerance)

            if index_min < 0 :
                index_min = len_rng_arr + index_min
                rng_arr = np.array(msg.ranges[index_min:len_rng_arr]+msg.ranges[0:index_max])

            elif index_max > len_rng_arr:
                index_max = index_max - len_rng_arr
                rng_arr = np.array(msg.ranges[index_min:len_rng_arr]+msg.ranges[0:(index_max-len_rng_arr)])
                
            else:
                rng_arr = np.array(msg.ranges[index_min:index_max])
                

            rng_arr =rng_arr[~np.isnan(rng_arr)]
            dist = float(np.mean(rng_arr))
            distance.append(dist)

        self.front_dist = distance[0] # 0.0
        self.leftfront_dist = distance[1] # 45.0
        self.left_dist = distance[2] #90.0
        self.right_dist = distance[3] # 270.0
        self.rightfront_dist = distance[4] # 325

    def print_debug_info(self):

        print("State: ", self.state)
        if self.current_class == 0:
           print("Empty Wall")
        elif self.current_class == 1:
            print("Left Turn")
        elif self.current_class == 2:
            print("Right Turn")
        elif self.current_class == 3 or self.current_class == 4:
           print("Stop")
        elif self.current_class == 5:
            print("Goal")
            

    def state_machine(self):

        self.print_debug_info()
       
        if self.state == "go":
            self.go_straight()
        elif self.state == "detection":
            # self.prev_class = self.current_class
            self.current_class = self.img_class
            self.state = "adjust_heading" 
            if self.current_class == 0:
                self.ang_setpoint = self.current_theta + np.pi / 4
            elif self.current_class == 1:
                self.ang_setpoint = self.current_theta + np.pi / 2
            elif self.current_class == 2:
                self.ang_setpoint =  self.current_theta - np.pi/2  
            elif self.current_class == 3 or self.current_theta == 4:
                self.ang_setpoint = self.current_theta + np.pi 
            elif self.current_class == 5:
                self.stop_flag = True
        elif self.state == "adjust_heading":
            self.adjust_heading()
        elif self.state == "action":
            if self.current_class == 0:
                self.action_empty_wall()
            elif self.current_class == 1:
                self.action_left_turn()
            elif self.current_class == 2:
                self.action_right_turn()
            elif self.current_class == 3 or self.current_class == 4:
                self.action_stop()
            elif self.current_class == 5:
                self.action_goal()


        if self.stop_flag:
            self.action_goal()
            print("Completed")

        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)


    def adjust_heading(self):

        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        if self.angle_wrt_lidar is not None:
            self.ang_err = self.angle_wrt_lidar - ANG_SETPT
            if abs(self.ang_err) > TOL:
                msg.angular.z = self.kp_ang_ah * self.ang_err 
                msg.linear.x  = 0.0
            else:
                self.state = "action"

        self.cmd_vel_pub.publish(msg)


    def callback_laser_scan(self, msg : Bool):
        if self.state == "go":
            if msg.data == True:
                self.state = "detection"

    def callback_object_dist(self, msg : Float64):
        self.object_dist = msg.data

    def callback_angle_lidar(self, msg):
        self.angle_wrt_lidar = msg.data
        

    def callback_img_class(self, msg : Int16):

        self.img_class = msg.data


    def callback_odom(self, Odom : Odometry):
        # print("Odom callback")
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_posx = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_posy = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_posz = position.z

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        self.current_x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_posx
        self.current_y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_posy
        self.current_theta = orientation - self.Init_ang
        

    def action_empty_wall(self): 
        """
            Rotate the Robot to find the next sign.
        """
    
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        
        theta_error = self.current_theta - self.ang_setpoint


        if theta_error <= -np.pi:
            theta_error += 2*np.pi
        elif theta_error > np.pi:
            theta_error -= 2*np.pi
        
        print("Last Theta: ", self.last_theta)
        print("Current Theta: ", self.current_theta)
        print("Theta Error :", theta_error)
        
        if math.fabs(theta_error) <= self.ang_tolerance or self.img_class != 0 :
            self.state = "go"
            print("State Change: ", self.state)
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)

        msg.angular.z =  self.kp_ang * abs(theta_error)  

        self.cmd_vel_pub.publish(msg)
        

    def action_left_turn(self):
        """
            Robot turns left by 90 deg.
            During this there is no symbols are detected.
            
        """

        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        
        theta_error = self.current_theta - self.ang_setpoint


        if theta_error <= -np.pi:
            theta_error += 2*np.pi
        elif theta_error > np.pi:
            theta_error -= 2*np.pi
        
        print("Last Theta: ", self.last_theta)
        print("Current Theta: ", self.current_theta)
        print("Theta Error :", theta_error)
        
        if math.fabs(theta_error) <= self.ang_tolerance :
            self.state = "go"
            print("State Change: ", self.state)
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)

        msg.angular.z =  self.kp_ang * abs(theta_error)  

        self.cmd_vel_pub.publish(msg)



    def action_right_turn(self):


        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        
        theta_error = self.current_theta - self.ang_setpoint


        if theta_error <= -np.pi:
            theta_error += 2*np.pi
        elif theta_error > np.pi:
            theta_error -= 2*np.pi
        
        print("Last Theta: ", self.last_theta)
        print("Current Theta: ", self.current_theta)
        print("Theta Error :", theta_error)
        
        if math.fabs(theta_error) <= self.ang_tolerance :
            self.state = "go"
            print("State Change: ", self.state)
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)

        msg.angular.z =  -1 * self.kp_ang * abs(theta_error)  

        self.cmd_vel_pub.publish(msg)

    def action_stop(self):
        """
            Robot Turns by 180 degrees
        """

        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        
        theta_error = self.current_theta - self.ang_setpoint


        if theta_error <= -np.pi:
            theta_error += 2*np.pi
        elif theta_error > np.pi:
            theta_error -= 2*np.pi
        
        print("Last Theta: ", self.last_theta)
        print("Current Theta: ", self.current_theta)
        print("Theta Error :", abs(theta_error))
        
        if math.fabs(theta_error) <= self.ang_tolerance :
            self.state = "go"
            print("State Change: ", self.state)
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)

        msg.angular.z =  self.kp_ang * abs(theta_error)  

        self.cmd_vel_pub.publish(msg)

        

    def action_goal(self):

        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)


    def go_straight(self): 
        """
            Rotate the Robot to find the next sign.
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        msg.linear.x = self.straight_speed

        self.cmd_vel_pub.publish(msg)


def main():
    rclpy.init() #init routine needed for ROS2.
    control_node = Controller()
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
