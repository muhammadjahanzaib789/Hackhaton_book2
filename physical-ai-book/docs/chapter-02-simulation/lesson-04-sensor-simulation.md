---
sidebar_position: 4
title: "Lesson 4: Sensor Simulation"
description: "Adding cameras, LIDAR, and IMU to simulated humanoids"
---

# Sensor Simulation

## Learning Objectives

By the end of this lesson, you will be able to:

1. Add various sensors to humanoid robots in Gazebo
2. Configure camera, LIDAR, and IMU sensors
3. Add realistic noise models to sensors
4. Publish sensor data to ROS 2 topics

## Prerequisites

- Completed Lessons 1-3 of this chapter
- Understanding of ROS 2 topics from Chapter 1
- Humanoid model loaded in Gazebo

## Sensor Overview for Humanoids

A perception-capable humanoid needs multiple sensors:

```
┌─────────────────────────────────────────────────────────────┐
│              Humanoid Sensor Configuration                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────┐                              │
│                    │  Head   │◀── RGB Camera                │
│                    │         │◀── Depth Camera              │
│                    └────┬────┘                              │
│                         │                                   │
│              ┌──────────┼──────────┐                       │
│              │          │          │                        │
│         Left │    ┌─────┴─────┐    │ Right                 │
│         Arm  │    │   Torso   │◀───┤ IMU                   │
│              │    │           │◀───┤ LIDAR                 │
│              │    └─────┬─────┘    │                       │
│              │          │          │                        │
│              ▼          │          ▼                        │
│         F/T Sensor      │      F/T Sensor                  │
│         (wrist)         │      (wrist)                     │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │                     │                        │
│         Left Leg              Right Leg                    │
│              │                     │                        │
│              ▼                     ▼                        │
│         F/T Sensor            F/T Sensor                   │
│         (ankle)               (ankle)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Sensor | Purpose | Typical Rate |
|--------|---------|--------------|
| **RGB Camera** | Object detection, navigation | 30 Hz |
| **Depth Camera** | 3D perception, obstacle avoidance | 15-30 Hz |
| **LIDAR** | Mapping, localization | 10-20 Hz |
| **IMU** | Balance, orientation | 100-400 Hz |
| **F/T Sensor** | Contact detection, manipulation | 100-1000 Hz |

## RGB Camera

### SDF Camera Configuration

```xml
<link name="camera_link">
  <pose relative_to="head">0.1 0 0.05 0 0 0</pose>

  <sensor name="head_camera" type="camera">
    <!-- Camera intrinsics -->
    <camera>
      <!-- Field of view (horizontal, radians) -->
      <horizontal_fov>1.396</horizontal_fov>

      <!-- Image properties -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>

      <!-- Clipping planes -->
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>

      <!-- Noise model -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>

      <!-- Distortion (optional) -->
      <distortion>
        <k1>0.0</k1>
        <k2>0.0</k2>
        <k3>0.0</k3>
        <p1>0.0</p1>
        <p2>0.0</p2>
        <center>0.5 0.5</center>
      </distortion>
    </camera>

    <!-- Sensor properties -->
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>

    <!-- ROS 2 topic -->
    <topic>/camera/image_raw</topic>
  </sensor>
</link>
```

### Camera Parameters Explained

| Parameter | Description | Humanoid Typical |
|-----------|-------------|------------------|
| `horizontal_fov` | Field of view (radians) | 1.0-1.5 (~60-90°) |
| `width/height` | Resolution | 640x480, 1280x720 |
| `format` | Pixel format | R8G8B8, L8 |
| `near/far` | Clipping distance | 0.1m to 100m |
| `update_rate` | Frames per second | 15-60 Hz |

### Camera Noise

Real cameras have noise. Adding noise in simulation improves sim-to-real transfer:

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.007</stddev>  <!-- Per-pixel standard deviation -->
</noise>
```

## Depth Camera

Depth cameras provide 3D information essential for manipulation and navigation.

### SDF Depth Camera Configuration

```xml
<link name="depth_camera_link">
  <pose relative_to="head">0.1 0 0 0 0 0</pose>

  <sensor name="depth_camera" type="depth_camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->

      <image>
        <width>640</width>
        <height>480</height>
        <format>R_FLOAT32</format>  <!-- Depth as float -->
      </image>

      <clip>
        <near>0.1</near>   <!-- Minimum range -->
        <far>10.0</far>    <!-- Maximum range -->
      </clip>

      <!-- Depth-specific noise -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
      </noise>
    </camera>

    <always_on>true</always_on>
    <update_rate>15</update_rate>
    <visualize>true</visualize>

    <topic>/depth_camera/depth/image_raw</topic>
  </sensor>
</link>
```

### Depth Camera Types

| Type | Range | Resolution | Use Case |
|------|-------|------------|----------|
| **Structured Light** | 0.5-4m | High | Indoor manipulation |
| **ToF** | 0.1-10m | Medium | Navigation |
| **Stereo** | 0.5-20m | Variable | Outdoor |

## LIDAR

LIDAR provides precise distance measurements for mapping and localization.

### SDF LIDAR Configuration

```xml
<link name="lidar_link">
  <pose relative_to="torso">0.15 0 0.3 0 0 0</pose>

  <sensor name="lidar" type="gpu_lidar">
    <lidar>
      <scan>
        <!-- Horizontal scan -->
        <horizontal>
          <samples>640</samples>
          <resolution>1</resolution>
          <min_angle>-2.356</min_angle>  <!-- -135 degrees -->
          <max_angle>2.356</max_angle>   <!-- +135 degrees -->
        </horizontal>

        <!-- Vertical scan (multi-line LIDAR) -->
        <vertical>
          <samples>16</samples>
          <resolution>1</resolution>
          <min_angle>-0.261</min_angle>  <!-- -15 degrees -->
          <max_angle>0.261</max_angle>   <!-- +15 degrees -->
        </vertical>
      </scan>

      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>

      <!-- Noise model -->
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </lidar>

    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>

    <topic>/scan</topic>
  </sensor>
</link>
```

### LIDAR Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    LIDAR Scan Pattern                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│        Top View (Single Scan Line)                          │
│                                                             │
│                    ▲ Robot Front                            │
│                    │                                        │
│               ╱    │    ╲                                  │
│           ╱        │        ╲                              │
│       ╱            │            ╲                          │
│    ╱ ─135°         │          +135° ╲                     │
│    ◀───────────────●───────────────▶                       │
│                  Robot                                      │
│                                                             │
│        Side View (Multi-Line)                               │
│                                                             │
│              ╱╱╱╱│╲╲╲╲                                    │
│           ╱╱╱╱  │  ╲╲╲╲                                   │
│        ═════════●═════════  16 lines                       │
│           ╲╲╲╲  │  ╱╱╱╱                                   │
│              ╲╲╲╲│╱╱╱╱                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## IMU (Inertial Measurement Unit)

IMU provides acceleration and angular velocity data critical for balance.

### SDF IMU Configuration

```xml
<link name="imu_link">
  <pose relative_to="torso">0 0 0 0 0 0</pose>

  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <visualize>true</visualize>

    <imu>
      <!-- Angular velocity noise -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>  <!-- rad/s -->
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0002</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>

      <!-- Linear acceleration noise -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>  <!-- m/s^2 -->
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>

    <topic>/imu/data</topic>
  </sensor>
</link>
```

### IMU Noise Model

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `stddev` (angular) | Noise standard deviation | 0.0002 rad/s |
| `stddev` (linear) | Noise standard deviation | 0.017 m/s² |
| `bias_mean` | Constant bias | Small offset |
| `bias_stddev` | Bias drift | Very small |

:::tip IMU Calibration
Real IMUs have bias that drifts over time. Including bias in simulation helps develop robust state estimation algorithms.
:::

## Force/Torque Sensors

Essential for contact detection and manipulation.

### SDF Force/Torque Configuration

```xml
<joint name="left_ankle_joint" type="revolute">
  <parent>left_shin</parent>
  <child>left_foot</child>

  <!-- Force/torque sensor -->
  <sensor name="left_ankle_ft" type="force_torque">
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>

    <always_on>true</always_on>
    <update_rate>100</update_rate>

    <topic>/left_foot/force_torque</topic>
  </sensor>
</joint>
```

### F/T Sensor Data

Output includes:
- Force: `(Fx, Fy, Fz)` in Newtons
- Torque: `(Tx, Ty, Tz)` in Newton-meters

```
┌─────────────────────────────────────────────────────────────┐
│              Force/Torque Sensor Output                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│        Fz ▲                                                 │
│           │      Tz (yaw)                                   │
│           │      ↺                                          │
│           │    ╱                                            │
│           │  ╱  Ty (pitch)                                 │
│           │╱    ↷                                          │
│    ◀──────┼──────▶ Fx                                      │
│    Fy   ╱ │                                                 │
│       ╱   │   Tx (roll)                                    │
│     ╱     │   ↻                                            │
│   ▼       │                                                 │
│                                                             │
│   Ground reaction force example (standing):                 │
│   Fz ≈ mass × gravity / 2 (per foot)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ROS 2 Bridge Configuration

To use sensor data in ROS 2, configure the Gazebo-ROS bridge.

### Bridge Launch File

```python
#!/usr/bin/env python3
"""
Gazebo ROS 2 Bridge Configuration
Physical AI Book - Chapter 2
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                # Camera
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',

                # Depth camera
                '/depth_camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',

                # LIDAR
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',

                # IMU
                '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',

                # Force/Torque
                '/left_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
                '/right_foot/force_torque@geometry_msgs/msg/Wrench[gz.msgs.Wrench',

                # Joint states
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',

                # Clock
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            ],
            output='screen',
        ),
    ])
```

## Sensor Data Subscriber Example

```python
#!/usr/bin/env python3
"""
Sensor Data Subscriber
Physical AI Book - Chapter 2

Demonstrates subscribing to simulated sensor data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import Wrench
from cv_bridge import CvBridge


class SensorSubscriber(Node):
    """Subscribes to all humanoid sensors."""

    def __init__(self):
        super().__init__('sensor_subscriber')

        self.bridge = CvBridge()

        # Camera subscription
        self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_callback, 10
        )

        # IMU subscription
        self.create_subscription(
            Imu, '/imu/data',
            self.imu_callback, 10
        )

        # LIDAR subscription
        self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, 10
        )

        # Force/Torque subscription
        self.create_subscription(
            Wrench, '/left_foot/force_torque',
            self.ft_callback, 10
        )

        self.get_logger().info('Sensor subscriber ready')

    def camera_callback(self, msg):
        """Process camera image."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info(
            f'Camera: {cv_image.shape[1]}x{cv_image.shape[0]}'
        )

    def imu_callback(self, msg):
        """Process IMU data."""
        self.get_logger().info(
            f'IMU: orientation=({msg.orientation.x:.3f}, '
            f'{msg.orientation.y:.3f}, {msg.orientation.z:.3f})'
        )

    def lidar_callback(self, msg):
        """Process LIDAR scan."""
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid_ranges:
            min_range = min(valid_ranges)
            self.get_logger().info(f'LIDAR: min_range={min_range:.2f}m')

    def ft_callback(self, msg):
        """Process force/torque data."""
        self.get_logger().info(
            f'F/T: force_z={msg.force.z:.1f}N'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SensorSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **Multiple sensors** are needed for humanoid perception
2. **Noise models** improve sim-to-real transfer
3. **Update rates** should match real sensor capabilities
4. **ROS 2 bridge** enables seamless integration
5. **Force/torque sensors** are critical for contact tasks

## Next Steps

In the [next lesson](./lesson-05-ros2-gazebo-bridge.md), we will:
- Configure the complete ROS 2-Gazebo bridge
- Implement joint controllers
- Create a full simulation launch system

## Additional Resources

- [Gazebo Sensors](https://gazebosim.org/docs/harmonic/sensors)
- [ROS 2 Sensor Messages](https://docs.ros.org/en/humble/p/sensor_msgs/)
- [Camera Calibration](https://docs.ros.org/en/humble/p/camera_calibration/)
