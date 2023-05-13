#!/usr/bin/env python

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription()

    package_name = 'hr13_lab6'

    sensor_node = Node(
        package=package_name,
        name='sensor_node',
        executable='sensor_node',
        # output="screen"
    )

    img_class = Node(
        package=package_name,
        name='img_class',
        executable='img_class',
        # output="screen"
    )

    controller = Node(
        package=package_name,
        name='controller',
        executable='controller',
        output="screen"
    )

    ld.add_action(sensor_node)
    ld.add_action(img_class)
    ld.add_action(controller)

    return ld

if __name__ == "__main__":
    generate_launch_description()
