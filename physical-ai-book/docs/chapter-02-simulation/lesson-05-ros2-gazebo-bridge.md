---
sidebar_position: 5
title: "Lesson 5: ROS 2 Gazebo Bridge"
description: "Integrating Gazebo Sim with ROS 2 for complete simulation"
---

# ROS 2 Gazebo Bridge

## Learning Objectives

By the end of this lesson, you will be able to:

1. Configure the ROS 2-Gazebo bridge for bidirectional communication
2. Implement joint position and velocity controllers
3. Create complete simulation launch files
4. Debug communication issues between ROS 2 and Gazebo

## Prerequisites

- Completed Lessons 1-4 of this chapter
- Understanding of ROS 2 topics and services
- Humanoid with sensors in Gazebo

## Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                ROS 2 ←→ Gazebo Bridge                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐      ┌─────────────────────┐     │
│  │      ROS 2          │      │     Gazebo Sim      │     │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │     │
│  │  │ /cmd_vel      │◀─┼──────┼──│ DiffDrive     │  │     │
│  │  │ /joint_states │◀─┼──────┼──│ JointState    │  │     │
│  │  │ /camera/image │◀─┼──────┼──│ Camera        │  │     │
│  │  │ /imu/data     │◀─┼──────┼──│ IMU           │  │     │
│  │  │ /scan         │◀─┼──────┼──│ LIDAR         │  │     │
│  │  └───────────────┘  │      │  └───────────────┘  │     │
│  │                     │      │                     │     │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │     │
│  │  │ /joint_cmd    │──┼──────┼─▶│ JointController│ │     │
│  │  │ /reset        │──┼──────┼─▶│ WorldControl  │  │     │
│  │  └───────────────┘  │      │  └───────────────┘  │     │
│  └─────────────────────┘      └─────────────────────┘     │
│                                                             │
│           ros_gz_bridge (parameter_bridge)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installing the Bridge

```bash
# Install ROS 2 Gazebo packages
sudo apt install ros-humble-ros-gz-bridge ros-humble-ros-gz-sim

# Verify installation
ros2 pkg list | grep ros_gz
```

## Bridge Configuration

### Basic Bridge Launch

```python
#!/usr/bin/env python3
"""
ROS 2 Gazebo Bridge Launch File
Physical AI Book - Chapter 2

Configures bidirectional communication between ROS 2 and Gazebo.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Bridge configuration
    bridge_params = [
        # Clock (essential for sim time)
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',

        # Joint states (Gazebo → ROS 2)
        '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',

        # Sensors (Gazebo → ROS 2)
        '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/depth_camera/depth@sensor_msgs/msg/Image[gz.msgs.Image',
        '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',

        # Force/torque sensors
        '/left_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
        '/right_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',

        # Commands (ROS 2 → Gazebo)
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',

        # Joint commands
        '/humanoid/joint_commands@std_msgs/msg/Float64MultiArray]gz.msgs.Double_V',
    ]

    return LaunchDescription([
        # Use simulation time
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),

        # Parameter bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=bridge_params,
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        # Image bridge for efficient image transport
        Node(
            package='ros_gz_image',
            executable='image_bridge',
            arguments=['/camera/image_raw'],
            output='screen',
        ),
    ])
```

### Bridge Syntax

The bridge uses a specific syntax:

```
/topic_name@ros_msg_type[gz_msg_type    # Gazebo → ROS 2
/topic_name@ros_msg_type]gz_msg_type    # ROS 2 → Gazebo
/topic_name@ros_msg_type@gz_msg_type    # Bidirectional
```

| Symbol | Direction | Example |
|--------|-----------|---------|
| `[` | Gazebo → ROS 2 | Sensor data |
| `]` | ROS 2 → Gazebo | Commands |
| `@` | Bidirectional | TF, pose |

## Joint Controllers

### Position Controller Plugin

Add to your SDF model:

```xml
<plugin filename="gz-sim-joint-position-controller-system"
        name="gz::sim::systems::JointPositionController">
  <joint_name>left_hip_pitch</joint_name>
  <topic>/humanoid/left_hip_pitch/cmd_pos</topic>
  <p_gain>100.0</p_gain>
  <i_gain>0.1</i_gain>
  <d_gain>10.0</d_gain>
  <i_max>1</i_max>
  <i_min>-1</i_min>
  <cmd_max>150</cmd_max>
  <cmd_min>-150</cmd_min>
</plugin>
```

### ROS 2 Joint Position Controller

```python
#!/usr/bin/env python3
"""
Joint Position Controller
Physical AI Book - Chapter 2

Sends position commands to humanoid joints via Gazebo.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import math


class JointPositionController(Node):
    """Controls humanoid joint positions."""

    # Joint names matching URDF/SDF
    JOINT_TOPICS = {
        'left_hip_pitch': '/humanoid/left_hip_pitch/cmd_pos',
        'left_knee_pitch': '/humanoid/left_knee_pitch/cmd_pos',
        'right_hip_pitch': '/humanoid/right_hip_pitch/cmd_pos',
        'right_knee_pitch': '/humanoid/right_knee_pitch/cmd_pos',
        # Add more joints...
    }

    def __init__(self):
        super().__init__('joint_position_controller')

        # Create publishers for each joint
        self.joint_pubs = {}
        for name, topic in self.JOINT_TOPICS.items():
            self.joint_pubs[name] = self.create_publisher(
                Float64, topic, 10
            )

        # Subscribe to joint states for feedback
        self.joint_states = {}
        self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10
        )

        # Timer for control loop
        self.create_timer(0.01, self.control_loop)  # 100 Hz

        # Target positions
        self.targets = {name: 0.0 for name in self.JOINT_TOPICS}

        self.get_logger().info('Joint controller ready')

    def joint_state_callback(self, msg):
        """Update current joint states."""
        for i, name in enumerate(msg.name):
            if name in self.joint_states:
                self.joint_states[name] = msg.position[i]

    def set_joint_position(self, joint_name: str, position: float):
        """
        Set target position for a joint.

        Args:
            joint_name: Name of the joint
            position: Target position in radians
        """
        if joint_name in self.targets:
            self.targets[joint_name] = position
        else:
            self.get_logger().warn(f'Unknown joint: {joint_name}')

    def control_loop(self):
        """Publish commands to joints."""
        for name, target in self.targets.items():
            msg = Float64()
            msg.data = target
            self.joint_pubs[name].publish(msg)

    def stand_pose(self):
        """Set humanoid to standing pose."""
        self.set_joint_position('left_hip_pitch', 0.0)
        self.set_joint_position('left_knee_pitch', 0.0)
        self.set_joint_position('right_hip_pitch', 0.0)
        self.set_joint_position('right_knee_pitch', 0.0)

    def squat_pose(self):
        """Set humanoid to squat pose."""
        self.set_joint_position('left_hip_pitch', -0.5)
        self.set_joint_position('left_knee_pitch', 1.0)
        self.set_joint_position('right_hip_pitch', -0.5)
        self.set_joint_position('right_knee_pitch', 1.0)


def main(args=None):
    rclpy.init(args=args)
    controller = JointPositionController()

    # Example: alternate between standing and squatting
    import time

    try:
        while rclpy.ok():
            controller.get_logger().info('Standing...')
            controller.stand_pose()
            rclpy.spin_once(controller, timeout_sec=2.0)
            time.sleep(2.0)

            controller.get_logger().info('Squatting...')
            controller.squat_pose()
            rclpy.spin_once(controller, timeout_sec=2.0)
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Complete Simulation Launch

### Full Launch File

```python
#!/usr/bin/env python3
"""
Complete Humanoid Simulation Launch
Physical AI Book - Chapter 2

Launches Gazebo, ROS 2 bridge, and all necessary nodes.
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Paths
    pkg_share = FindPackageShare('physical_ai_book')
    world_path = PathJoinSubstitution([pkg_share, 'worlds', 'humanoid_world.sdf'])
    urdf_path = PathJoinSubstitution([pkg_share, 'models', 'humanoid', 'humanoid.urdf'])

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gui = LaunchConfiguration('gui', default='true')
    headless = LaunchConfiguration('headless', default='false')

    # Set Gazebo resource paths
    gz_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_share, 'static', 'models')
    )

    # Gazebo simulation
    gz_sim = ExecuteProcess(
        cmd=[
            FindExecutable(name='gz'),
            'sim',
            '-r',  # Run immediately
            LaunchConfiguration('world', default=world_path),
        ],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['cat ', urdf_path]),
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Spawn humanoid in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'humanoid',
            '-file', urdf_path,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.95',
        ],
        output='screen',
    )

    # RViz2 (optional)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'humanoid.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(gui),
    )

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('world', default_value=world_path),

        # Environment
        gz_resource_path,

        # Nodes
        gz_sim,
        robot_state_publisher,
        bridge,
        spawn_robot,
        rviz,
    ])
```

### Running the Simulation

```bash
# Build the package
cd physical-ai-book
colcon build

# Source the workspace
source install/setup.bash

# Launch the simulation
ros2 launch physical_ai_book humanoid_sim.launch.py

# In another terminal, check topics
ros2 topic list

# Expected output:
# /camera/camera_info
# /camera/image_raw
# /clock
# /cmd_vel
# /imu/data
# /joint_states
# /parameter_events
# /robot_description
# /rosout
# /scan
# /tf
# /tf_static
```

## Debugging Bridge Issues

### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| No topics visible | Bridge not running | Check bridge node |
| Wrong message type | Syntax error | Verify `[` vs `]` |
| No clock | Clock not bridged | Add clock to bridge |
| Stale data | `use_sim_time` wrong | Set to `true` |

### Debugging Commands

```bash
# Check if bridge is running
ros2 node list | grep bridge

# Inspect a topic
ros2 topic info /camera/image_raw

# Check message frequency
ros2 topic hz /imu/data

# Echo messages
ros2 topic echo /joint_states --once

# Check Gazebo topics (in Gazebo)
gz topic -l
```

### Bridge Health Check

```python
#!/usr/bin/env python3
"""
Bridge Health Check
Physical AI Book - Chapter 2

Verifies ROS 2-Gazebo bridge is working correctly.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from rosgraph_msgs.msg import Clock


class BridgeHealthCheck(Node):
    """Checks that all bridged topics are receiving data."""

    REQUIRED_TOPICS = {
        '/clock': (Clock, 'Clock'),
        '/joint_states': (JointState, 'Joint States'),
        '/imu/data': (Imu, 'IMU'),
        '/camera/image_raw': (Image, 'Camera'),
        '/scan': (LaserScan, 'LIDAR'),
    }

    def __init__(self):
        super().__init__('bridge_health_check')

        self.received = {topic: False for topic in self.REQUIRED_TOPICS}

        # Subscribe to all topics
        for topic, (msg_type, name) in self.REQUIRED_TOPICS.items():
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self.mark_received(t),
                10
            )

        # Check timer
        self.create_timer(5.0, self.report_status)

        self.get_logger().info('Bridge health check started')

    def mark_received(self, topic):
        """Mark a topic as received."""
        if not self.received[topic]:
            self.received[topic] = True
            name = self.REQUIRED_TOPICS[topic][1]
            self.get_logger().info(f'✓ {name} ({topic})')

    def report_status(self):
        """Report overall bridge status."""
        missing = [t for t, r in self.received.items() if not r]

        if not missing:
            self.get_logger().info('✓ All topics active - bridge healthy!')
        else:
            self.get_logger().warn(f'✗ Missing topics: {missing}')


def main(args=None):
    rclpy.init(args=args)
    node = BridgeHealthCheck()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **ros_gz_bridge** enables ROS 2-Gazebo communication
2. **Bridge syntax** (`[`, `]`, `@`) controls data direction
3. **Joint controllers** require proper PID tuning
4. **Complete launch files** orchestrate all components
5. **use_sim_time** must be consistent across all nodes

## Next Steps

In the [exercises](./exercises.md), you will:
- Build a complete humanoid simulation from scratch
- Implement basic locomotion control
- Create a sensor fusion node

## Additional Resources

- [ros_gz Documentation](https://github.com/gazebosim/ros_gz)
- [Gazebo ROS 2 Integration](https://gazebosim.org/docs/harmonic/ros2_integration)
- [ROS 2 Control](https://control.ros.org/master/index.html)
