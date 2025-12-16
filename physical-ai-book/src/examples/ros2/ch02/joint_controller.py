#!/usr/bin/env python3
"""
Joint Position Controller
Physical AI Book - Chapter 2: Simulation

Demonstrates position control of humanoid joints via Gazebo bridge.
Provides interfaces for setting individual joint positions and
pre-defined poses.

Usage:
    # Start simulation first
    ros2 launch physical_ai_examples launch_humanoid.py

    # Run controller
    ros2 run physical_ai_examples joint_controller

Expected Output:
    [INFO] [joint_controller]: Joint controller ready
    [INFO] [joint_controller]: Setting pose: stand
    [INFO] [joint_controller]: Joint states received: 21 joints

Dependencies:
    - rclpy
    - std_msgs
    - sensor_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from typing import Dict, List, Optional
import math


class JointController(Node):
    """
    Position controller for humanoid robot joints.

    Publishes:
        /humanoid/{joint_name}/cmd_pos (Float64): Position commands

    Subscribes:
        /joint_states (JointState): Current joint positions

    This demonstrates:
    - Publishing to Gazebo joint controllers
    - Subscribing to joint state feedback
    - Implementing pre-defined robot poses
    """

    # Joint configuration matching URDF
    JOINT_NAMES = [
        # Torso
        'waist_yaw',
        # Left arm (7 DOF)
        'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
        'left_elbow_pitch',
        'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll',
        # Right arm (7 DOF)
        'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
        'right_elbow_pitch',
        'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll',
        # Left leg (6 DOF)
        'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
        'left_knee_pitch',
        'left_ankle_pitch', 'left_ankle_roll',
        # Right leg (6 DOF)
        'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
        'right_knee_pitch',
        'right_ankle_pitch', 'right_ankle_roll',
    ]

    # Pre-defined poses
    POSES = {
        'stand': {
            'waist_yaw': 0.0,
            'left_shoulder_pitch': 0.0, 'left_shoulder_roll': 0.1,
            'left_elbow_pitch': 0.0,
            'right_shoulder_pitch': 0.0, 'right_shoulder_roll': -0.1,
            'right_elbow_pitch': 0.0,
            'left_hip_pitch': 0.0, 'left_knee_pitch': 0.0,
            'left_ankle_pitch': 0.0,
            'right_hip_pitch': 0.0, 'right_knee_pitch': 0.0,
            'right_ankle_pitch': 0.0,
        },
        'squat': {
            'left_hip_pitch': -0.6, 'left_knee_pitch': 1.2,
            'left_ankle_pitch': -0.6,
            'right_hip_pitch': -0.6, 'right_knee_pitch': 1.2,
            'right_ankle_pitch': -0.6,
        },
        'arms_up': {
            'left_shoulder_pitch': -1.5, 'left_elbow_pitch': 0.0,
            'right_shoulder_pitch': -1.5, 'right_elbow_pitch': 0.0,
        },
        'arms_out': {
            'left_shoulder_roll': 1.5, 'left_elbow_pitch': 0.0,
            'right_shoulder_roll': -1.5, 'right_elbow_pitch': 0.0,
        },
        't_pose': {
            'left_shoulder_roll': 1.57, 'left_elbow_pitch': 0.0,
            'right_shoulder_roll': -1.57, 'right_elbow_pitch': 0.0,
            'left_hip_pitch': 0.0, 'left_knee_pitch': 0.0,
            'right_hip_pitch': 0.0, 'right_knee_pitch': 0.0,
        },
    }

    def __init__(self):
        super().__init__('joint_controller')

        # Create publishers for each joint
        self.joint_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        for joint_name in self.JOINT_NAMES:
            topic = f'/humanoid/{joint_name}/cmd_pos'
            self.joint_pubs[joint_name] = self.create_publisher(
                Float64, topic, 10
            )

        # Subscribe to joint states for feedback
        self.current_positions: Dict[str, float] = {}
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Target positions
        self.targets: Dict[str, float] = {name: 0.0 for name in self.JOINT_NAMES}

        # Control loop (50 Hz)
        self.create_timer(0.02, self.control_loop)

        self.get_logger().info('Joint controller ready')

    def joint_state_callback(self, msg: JointState):
        """
        Process joint state feedback.

        Args:
            msg: JointState message with current positions
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

        # Log occasionally
        if not hasattr(self, '_log_count'):
            self._log_count = 0
        self._log_count += 1
        if self._log_count % 100 == 0:
            self.get_logger().debug(
                f'Joint states received: {len(msg.name)} joints'
            )

    def control_loop(self):
        """Publish target positions to joints."""
        for joint_name, target in self.targets.items():
            if joint_name in self.joint_pubs:
                msg = Float64()
                msg.data = target
                self.joint_pubs[joint_name].publish(msg)

    def set_joint_position(self, joint_name: str, position: float):
        """
        Set target position for a single joint.

        Args:
            joint_name: Name of the joint
            position: Target position in radians
        """
        if joint_name in self.targets:
            self.targets[joint_name] = position
            self.get_logger().debug(f'Set {joint_name} to {position:.3f} rad')
        else:
            self.get_logger().warn(f'Unknown joint: {joint_name}')

    def set_pose(self, pose_name: str):
        """
        Set robot to a pre-defined pose.

        Args:
            pose_name: Name of the pose (stand, squat, arms_up, etc.)
        """
        if pose_name not in self.POSES:
            self.get_logger().error(f'Unknown pose: {pose_name}')
            self.get_logger().info(f'Available poses: {list(self.POSES.keys())}')
            return

        self.get_logger().info(f'Setting pose: {pose_name}')
        pose = self.POSES[pose_name]
        for joint_name, position in pose.items():
            self.set_joint_position(joint_name, position)

    def get_joint_position(self, joint_name: str) -> Optional[float]:
        """
        Get current position of a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Current position in radians, or None if unknown
        """
        return self.current_positions.get(joint_name)

    def interpolate_to_pose(
        self,
        pose_name: str,
        duration: float = 2.0,
        steps: int = 50
    ):
        """
        Smoothly interpolate to a target pose.

        Args:
            pose_name: Target pose name
            duration: Time to reach pose in seconds
            steps: Number of interpolation steps

        Note: This is a blocking call.
        """
        if pose_name not in self.POSES:
            self.get_logger().error(f'Unknown pose: {pose_name}')
            return

        target_pose = self.POSES[pose_name]
        start_positions = dict(self.targets)

        dt = duration / steps
        for step in range(steps + 1):
            t = step / steps  # 0 to 1
            # Smooth interpolation (ease in-out)
            t = t * t * (3 - 2 * t)

            for joint_name, target in target_pose.items():
                start = start_positions.get(joint_name, 0.0)
                current = start + t * (target - start)
                self.targets[joint_name] = current

            # Sleep for interpolation timing
            import time
            time.sleep(dt)

        self.get_logger().info(f'Reached pose: {pose_name}')


def main(args=None):
    """
    Main entry point.

    Demonstrates pose cycling.
    """
    rclpy.init(args=args)

    controller = JointController()

    # Demo: cycle through poses
    import time
    poses = ['stand', 't_pose', 'arms_up', 'squat', 'stand']

    try:
        for pose in poses:
            controller.set_pose(pose)
            controller.get_logger().info(f'Holding pose: {pose}')

            # Spin for 3 seconds
            start = time.time()
            while time.time() - start < 3.0:
                rclpy.spin_once(controller, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
