#!/usr/bin/env python3
"""
Multi-Sensor Reader
Physical AI Book - Chapter 2: Simulation

Subscribes to all humanoid sensors and demonstrates data processing.
Shows how to handle camera, LIDAR, IMU, and force/torque data.

Usage:
    # Start simulation first
    ros2 launch physical_ai_examples launch_humanoid.py

    # Run sensor reader
    ros2 run physical_ai_examples sensor_reader

Expected Output:
    [INFO] [sensor_reader]: Sensor reader initialized
    [INFO] [sensor_reader]: Camera: 640x480 @ 30Hz
    [INFO] [sensor_reader]: IMU: roll=0.00° pitch=0.00° yaw=0.00°
    [INFO] [sensor_reader]: LIDAR: 640 points, min=0.5m, max=10.0m
    [INFO] [sensor_reader]: Left foot: Fz=49.1N (contact)

Dependencies:
    - rclpy
    - sensor_msgs
    - geometry_msgs
    - cv_bridge (optional, for image conversion)

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan, JointState
from geometry_msgs.msg import Wrench
import math
from typing import Optional


class SensorReader(Node):
    """
    Multi-sensor subscriber for humanoid robot.

    Subscribes:
        /camera/image_raw (Image): RGB camera
        /imu/data (Imu): Inertial measurement unit
        /scan (LaserScan): LIDAR scanner
        /joint_states (JointState): Joint positions
        /left_foot/force_torque (Wrench): Left foot contact
        /right_foot/force_torque (Wrench): Right foot contact

    This demonstrates:
    - Subscribing to multiple sensor topics
    - Processing different message types
    - Extracting useful information from raw data
    """

    # Contact force threshold (Newtons)
    CONTACT_THRESHOLD = 10.0

    def __init__(self):
        super().__init__('sensor_reader')

        # === Sensor data storage ===
        self.camera_info = {'width': 0, 'height': 0, 'count': 0}
        self.imu_orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.lidar_stats = {'points': 0, 'min': 0.0, 'max': 0.0}
        self.foot_forces = {'left': 0.0, 'right': 0.0}
        self.joint_count = 0

        # === Subscriptions ===
        # Camera
        self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_callback, 10
        )

        # IMU
        self.create_subscription(
            Imu, '/imu/data',
            self.imu_callback, 10
        )

        # LIDAR
        self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, 10
        )

        # Joint states
        self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )

        # Force/torque sensors
        self.create_subscription(
            Wrench, '/left_foot/force_torque',
            lambda msg: self.foot_callback(msg, 'left'), 10
        )
        self.create_subscription(
            Wrench, '/right_foot/force_torque',
            lambda msg: self.foot_callback(msg, 'right'), 10
        )

        # Status reporting timer (1 Hz)
        self.create_timer(1.0, self.report_status)

        self.get_logger().info('Sensor reader initialized')

    def camera_callback(self, msg: Image):
        """
        Process camera image.

        Args:
            msg: Image message
        """
        self.camera_info['width'] = msg.width
        self.camera_info['height'] = msg.height
        self.camera_info['count'] += 1

        # For actual image processing, use cv_bridge:
        # from cv_bridge import CvBridge
        # bridge = CvBridge()
        # cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')

    def imu_callback(self, msg: Imu):
        """
        Process IMU data and extract orientation.

        Args:
            msg: Imu message with orientation quaternion
        """
        # Convert quaternion to Euler angles
        q = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(q.x, q.y, q.z, q.w)

        self.imu_orientation['roll'] = math.degrees(roll)
        self.imu_orientation['pitch'] = math.degrees(pitch)
        self.imu_orientation['yaw'] = math.degrees(yaw)

    def lidar_callback(self, msg: LaserScan):
        """
        Process LIDAR scan data.

        Args:
            msg: LaserScan message
        """
        # Filter valid ranges
        valid_ranges = [
            r for r in msg.ranges
            if msg.range_min < r < msg.range_max
        ]

        if valid_ranges:
            self.lidar_stats['points'] = len(valid_ranges)
            self.lidar_stats['min'] = min(valid_ranges)
            self.lidar_stats['max'] = max(valid_ranges)
        else:
            self.lidar_stats['points'] = 0
            self.lidar_stats['min'] = 0.0
            self.lidar_stats['max'] = 0.0

    def joint_callback(self, msg: JointState):
        """
        Process joint state data.

        Args:
            msg: JointState message
        """
        self.joint_count = len(msg.name)

    def foot_callback(self, msg: Wrench, foot: str):
        """
        Process force/torque sensor data.

        Args:
            msg: Wrench message
            foot: 'left' or 'right'
        """
        # Store vertical force (Z component)
        self.foot_forces[foot] = msg.force.z

    def report_status(self):
        """Log current sensor status."""
        # Camera status
        if self.camera_info['count'] > 0:
            self.get_logger().info(
                f"Camera: {self.camera_info['width']}x{self.camera_info['height']}"
            )

        # IMU status
        self.get_logger().info(
            f"IMU: roll={self.imu_orientation['roll']:.1f}° "
            f"pitch={self.imu_orientation['pitch']:.1f}° "
            f"yaw={self.imu_orientation['yaw']:.1f}°"
        )

        # LIDAR status
        if self.lidar_stats['points'] > 0:
            self.get_logger().info(
                f"LIDAR: {self.lidar_stats['points']} points, "
                f"min={self.lidar_stats['min']:.2f}m, "
                f"max={self.lidar_stats['max']:.2f}m"
            )

        # Foot contact status
        for foot in ['left', 'right']:
            force = self.foot_forces[foot]
            contact = 'contact' if abs(force) > self.CONTACT_THRESHOLD else 'air'
            self.get_logger().info(
                f"{foot.capitalize()} foot: Fz={force:.1f}N ({contact})"
            )

        # Joint status
        if self.joint_count > 0:
            self.get_logger().info(f"Joints: {self.joint_count} active")

    @staticmethod
    def quaternion_to_euler(x: float, y: float, z: float, w: float):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).

        Args:
            x, y, z, w: Quaternion components

        Returns:
            tuple: (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def is_standing(self) -> bool:
        """
        Check if robot has both feet on the ground.

        Returns:
            True if both feet are in contact
        """
        return (
            abs(self.foot_forces['left']) > self.CONTACT_THRESHOLD and
            abs(self.foot_forces['right']) > self.CONTACT_THRESHOLD
        )

    def is_tilted(self, threshold: float = 15.0) -> bool:
        """
        Check if robot is significantly tilted.

        Args:
            threshold: Tilt threshold in degrees

        Returns:
            True if roll or pitch exceeds threshold
        """
        return (
            abs(self.imu_orientation['roll']) > threshold or
            abs(self.imu_orientation['pitch']) > threshold
        )


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    reader = SensorReader()

    try:
        rclpy.spin(reader)
    except KeyboardInterrupt:
        pass

    reader.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
