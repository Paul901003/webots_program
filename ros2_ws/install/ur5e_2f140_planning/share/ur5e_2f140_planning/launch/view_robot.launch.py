import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("ur5e_2f140_planning")
    urdf_path = os.path.join(pkg, "urdf", "ur5e_2f140.urdf.xacro")
    robot_description = xacro.process_file(urdf_path).toxml()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg, "config", "view_robot.rviz")],
        ),
    ])
