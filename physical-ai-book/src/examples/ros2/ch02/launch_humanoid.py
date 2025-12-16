#!/usr/bin/env python3
"""
Humanoid Launch Configuration
Physical AI Book - Chapter 2: Simulation

Launch file to start Gazebo Sim with humanoid robot and ROS 2 bridge.

Usage:
    ros2 launch physical_ai_examples launch_humanoid.py

    # With custom world
    ros2 launch physical_ai_examples launch_humanoid.py world:=custom_world.sdf

    # Headless mode (no GUI)
    ros2 launch physical_ai_examples launch_humanoid.py gui:=false

Expected Output:
    [INFO] [gz_sim]: Starting Gazebo Sim...
    [INFO] [ros_gz_bridge]: Bridge configured
    [INFO] [robot_state_publisher]: Publishing robot description

Dependencies:
    - ros2
    - gazebo (gz-sim)
    - ros_gz_bridge
    - robot_state_publisher

Author: Physical AI Book
License: MIT
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for humanoid simulation."""

    # === Paths ===
    # Adjust these paths for your installation
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))
    models_path = os.path.join(pkg_path, 'static', 'models')
    worlds_path = os.path.join(pkg_path, 'worlds')
    urdf_path = os.path.join(models_path, 'humanoid', 'humanoid.urdf')

    # === Launch Arguments ===
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui', default='true')
    world = LaunchConfiguration('world', default='humanoid_world.sdf')
    x_pose = LaunchConfiguration('x', default='0.0')
    y_pose = LaunchConfiguration('y', default='0.0')
    z_pose = LaunchConfiguration('z', default='0.95')

    # === Environment Variables ===
    gz_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        models_path
    )

    # === Gazebo Sim ===
    # Construct the gz sim command
    gz_sim_cmd = [
        FindExecutable(name='gz'),
        'sim',
        '-r',  # Run immediately
    ]

    gz_sim = ExecuteProcess(
        cmd=gz_sim_cmd + [world],
        output='screen',
        additional_env={'GZ_SIM_RESOURCE_PATH': models_path},
    )

    # === ROS 2 - Gazebo Bridge ===
    bridge_params = [
        # Clock (essential)
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

        # Joint states
        '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',

        # Sensors
        '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/depth_camera/depth@sensor_msgs/msg/Image[gz.msgs.Image',
        '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',

        # Force/torque
        '/left_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
        '/right_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',

        # Commands
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
    ]

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_params,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # === Robot State Publisher ===
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['cat ', urdf_path]),
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # === Spawn Robot ===
    # Delay spawn to ensure Gazebo is ready
    spawn_robot = TimerAction(
        period=3.0,  # Wait 3 seconds
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'humanoid',
                    '-file', urdf_path,
                    '-x', x_pose,
                    '-y', y_pose,
                    '-z', z_pose,
                ],
                output='screen',
            )
        ]
    )

    # === Launch Description ===
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Launch Gazebo with GUI'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='humanoid_world.sdf',
            description='World file to load'
        ),
        DeclareLaunchArgument('x', default_value='0.0', description='X position'),
        DeclareLaunchArgument('y', default_value='0.0', description='Y position'),
        DeclareLaunchArgument('z', default_value='0.95', description='Z position'),

        # Environment
        gz_resource_path,

        # Nodes
        gz_sim,
        ros_gz_bridge,
        robot_state_publisher,
        spawn_robot,
    ])


if __name__ == '__main__':
    # Allow running as standalone script for testing
    from launch import LaunchService
    ls = LaunchService()
    ls.include_launch_description(generate_launch_description())
    ls.run()
