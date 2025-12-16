#!/usr/bin/env python3
"""
Simple Locomotion Controller
Physical AI Book - Chapter 2: Simulation

Implements basic open-loop walking motion for the humanoid robot.
Uses sinusoidal joint trajectories to generate a walking gait.

Usage:
    # Start simulation first
    ros2 launch physical_ai_examples launch_humanoid.py

    # Run locomotion controller
    ros2 run physical_ai_examples simple_locomotion

    # With custom gait parameters
    ros2 run physical_ai_examples simple_locomotion --ros-args \
        -p frequency:=0.5 -p hip_amplitude:=0.3

Expected Output:
    [INFO] [simple_locomotion]: Locomotion controller started
    [INFO] [simple_locomotion]: Gait: freq=0.5Hz, hip=0.25rad, knee=0.40rad
    [INFO] [simple_locomotion]: Walking...
    [INFO] [simple_locomotion]: Step count: 10

Dependencies:
    - rclpy
    - std_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math


class SimpleLocomotion(Node):
    """
    Open-loop walking controller using sinusoidal joint trajectories.

    Publishes:
        /humanoid/{joint_name}/cmd_pos (Float64): Position commands

    Parameters:
        frequency (float): Gait frequency in Hz (default: 0.5)
        hip_amplitude (float): Hip swing amplitude in radians (default: 0.25)
        knee_amplitude (float): Knee bend amplitude in radians (default: 0.4)
        ankle_amplitude (float): Ankle compensation in radians (default: 0.15)

    This demonstrates:
    - Central Pattern Generator (CPG) concept
    - Phase-coordinated leg movement
    - Sinusoidal trajectory generation
    """

    def __init__(self):
        super().__init__('simple_locomotion')

        # === Declare parameters ===
        self.declare_parameter('frequency', 0.5)
        self.declare_parameter('hip_amplitude', 0.25)
        self.declare_parameter('knee_amplitude', 0.40)
        self.declare_parameter('ankle_amplitude', 0.15)
        self.declare_parameter('arm_swing', 0.3)

        # Get parameters
        self.frequency = self.get_parameter('frequency').value
        self.hip_amp = self.get_parameter('hip_amplitude').value
        self.knee_amp = self.get_parameter('knee_amplitude').value
        self.ankle_amp = self.get_parameter('ankle_amplitude').value
        self.arm_swing = self.get_parameter('arm_swing').value

        # Phase offset between legs (180 degrees for alternating gait)
        self.phase_offset = math.pi

        # Current gait phase (0 to 2π)
        self.phase = 0.0

        # Step counter
        self.step_count = 0

        # === Create joint publishers ===
        self.pubs = {}

        # Leg joints
        leg_joints = [
            'left_hip_pitch', 'left_knee_pitch', 'left_ankle_pitch',
            'right_hip_pitch', 'right_knee_pitch', 'right_ankle_pitch',
        ]

        # Arm joints (for natural arm swing)
        arm_joints = [
            'left_shoulder_pitch', 'right_shoulder_pitch',
        ]

        for joint in leg_joints + arm_joints:
            topic = f'/humanoid/{joint}/cmd_pos'
            self.pubs[joint] = self.create_publisher(Float64, topic, 10)

        # === Control loop ===
        self.control_rate = 100  # Hz
        self.create_timer(1.0 / self.control_rate, self.control_loop)

        # Log configuration
        self.get_logger().info('Locomotion controller started')
        self.get_logger().info(
            f'Gait: freq={self.frequency}Hz, '
            f'hip={self.hip_amp}rad, knee={self.knee_amp}rad'
        )

    def control_loop(self):
        """Generate and publish walking motion."""
        # Update phase
        dt = 1.0 / self.control_rate
        self.phase += 2 * math.pi * self.frequency * dt

        # Wrap phase to [0, 2π]
        if self.phase >= 2 * math.pi:
            self.phase -= 2 * math.pi
            self.step_count += 1
            if self.step_count % 10 == 0:
                self.get_logger().info(f'Step count: {self.step_count}')

        # Calculate joint positions for each leg
        left_joints = self.calculate_leg_joints(self.phase)
        right_joints = self.calculate_leg_joints(self.phase + self.phase_offset)

        # Publish leg joints
        self.publish_joint('left_hip_pitch', left_joints['hip'])
        self.publish_joint('left_knee_pitch', left_joints['knee'])
        self.publish_joint('left_ankle_pitch', left_joints['ankle'])

        self.publish_joint('right_hip_pitch', right_joints['hip'])
        self.publish_joint('right_knee_pitch', right_joints['knee'])
        self.publish_joint('right_ankle_pitch', right_joints['ankle'])

        # Natural arm swing (opposite to legs)
        arm_left = self.arm_swing * math.sin(self.phase + self.phase_offset)
        arm_right = self.arm_swing * math.sin(self.phase)
        self.publish_joint('left_shoulder_pitch', arm_left)
        self.publish_joint('right_shoulder_pitch', arm_right)

    def calculate_leg_joints(self, phase: float) -> dict:
        """
        Calculate joint positions for one leg at a given phase.

        The gait cycle:
        - phase 0 to π: Swing phase (leg moving forward)
        - phase π to 2π: Stance phase (leg supporting body)

        Args:
            phase: Current gait phase (0 to 2π)

        Returns:
            dict: Joint positions {hip, knee, ankle}
        """
        # Hip: sinusoidal forward/backward swing
        hip = self.hip_amp * math.sin(phase)

        # Knee: bend during swing phase (phase 0 to π)
        # Use rectified sine for one-sided bending
        knee = self.knee_amp * max(0, math.sin(phase))

        # Ankle: compensates for hip angle to keep foot level
        # Delayed phase for ground clearance timing
        ankle = -self.ankle_amp * math.sin(phase - math.pi / 4)

        return {'hip': hip, 'knee': knee, 'ankle': ankle}

    def publish_joint(self, joint_name: str, position: float):
        """
        Publish position command to a joint.

        Args:
            joint_name: Name of the joint
            position: Target position in radians
        """
        msg = Float64()
        msg.data = position
        self.pubs[joint_name].publish(msg)

    def stop(self):
        """Stop walking and return to neutral pose."""
        self.get_logger().info('Stopping - returning to neutral')

        neutral = {
            'left_hip_pitch': 0.0, 'left_knee_pitch': 0.0, 'left_ankle_pitch': 0.0,
            'right_hip_pitch': 0.0, 'right_knee_pitch': 0.0, 'right_ankle_pitch': 0.0,
            'left_shoulder_pitch': 0.0, 'right_shoulder_pitch': 0.0,
        }

        for joint, pos in neutral.items():
            self.publish_joint(joint, pos)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    controller = SimpleLocomotion()

    try:
        controller.get_logger().info('Walking... Press Ctrl+C to stop')
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.stop()

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
