import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("ur5e_2f140", package_name="ur5e_2f140_planning")
        .robot_description(file_path="urdf/ur5e_2f140.urdf.xacro")
        .robot_description_semantic(file_path="config/ur5e_2f140.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    planning_bridge = Node(
        package="ur5e_2f140_planning",
        executable="planning_bridge",
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        move_group,
        planning_bridge,
    ])
