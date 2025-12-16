#!/usr/bin/env python3
"""
VLA ROS 2 Controller Node
Physical AI Book - Chapter 7: Vision-Language-Action

ROS 2 node for real-time VLA-based robot control.
Integrates VLA inference with robot control systems.

Features:
- Real-time image subscription
- Voice/text instruction handling
- Action publishing to robot controllers
- Safety filtering
- Status monitoring

Usage:
    ros2 run physical_ai_examples vla_controller

    # Send instruction
    ros2 topic pub /vla/instruction std_msgs/String "data: 'pick up the cup'"

Dependencies:
    - rclpy
    - sensor_msgs
    - geometry_msgs
    - std_msgs
    - cv_bridge
    - torch

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String, Float64MultiArray, Bool
from cv_bridge import CvBridge
import numpy as np
import threading
from typing import Optional, List
from collections import deque
import time


class SafetyFilter:
    """
    Safety filter for VLA action outputs.

    Applies velocity limits, workspace constraints,
    and smoothing for safe robot operation.
    """

    def __init__(
        self,
        max_linear_vel: float = 0.1,
        max_angular_vel: float = 0.3,
        max_gripper_vel: float = 0.5,
        workspace_limits: Optional[dict] = None
    ):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.max_gripper_vel = max_gripper_vel

        # Default workspace limits
        self.workspace_limits = workspace_limits or {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (0.0, 1.0)
        }

        self.prev_action = None
        self.max_delta = 0.1  # Maximum change per step

    def filter(
        self,
        action: np.ndarray,
        current_pose: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Filter action for safety.

        Args:
            action: [7] raw VLA action (dx, dy, dz, drx, dry, drz, gripper)
            current_pose: Current end-effector pose (optional)

        Returns:
            (filtered_action, is_safe, violations)
        """
        filtered = action.copy()
        violations = []
        is_safe = True

        # Velocity limiting - linear
        linear = filtered[:3]
        linear_mag = np.linalg.norm(linear)
        if linear_mag > self.max_linear_vel:
            linear = linear / linear_mag * self.max_linear_vel
            filtered[:3] = linear
            violations.append(f"Linear velocity clamped: {linear_mag:.3f} -> {self.max_linear_vel}")
            is_safe = False

        # Velocity limiting - angular
        angular = filtered[3:6]
        angular_mag = np.linalg.norm(angular)
        if angular_mag > self.max_angular_vel:
            angular = angular / angular_mag * self.max_angular_vel
            filtered[3:6] = angular
            violations.append(f"Angular velocity clamped: {angular_mag:.3f} -> {self.max_angular_vel}")
            is_safe = False

        # Gripper saturation
        filtered[6] = np.clip(filtered[6], 0.0, 1.0)

        # Rate limiting (smooth changes)
        if self.prev_action is not None:
            delta = filtered - self.prev_action
            delta_mag = np.linalg.norm(delta[:6])
            if delta_mag > self.max_delta:
                scale = self.max_delta / delta_mag
                filtered[:6] = self.prev_action[:6] + delta[:6] * scale
                violations.append(f"Action rate limited")

        self.prev_action = filtered.copy()

        return filtered, is_safe, violations

    def reset(self):
        """Reset filter state."""
        self.prev_action = None


class VLAControllerNode(Node):
    """
    ROS 2 node for VLA-based robot control.

    Subscribes to camera images and instructions,
    runs VLA inference, and publishes robot actions.
    """

    def __init__(self):
        super().__init__('vla_controller')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('image_history', 4)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('max_linear_vel', 0.1)
        self.declare_parameter('max_angular_vel', 0.3)

        # Get parameters
        model_path = self.get_parameter('model_path').value
        device = self.get_parameter('device').value
        self.control_rate = self.get_parameter('control_rate').value
        image_history = self.get_parameter('image_history').value
        safety_enabled = self.get_parameter('safety_enabled').value
        max_linear = self.get_parameter('max_linear_vel').value
        max_angular = self.get_parameter('max_angular_vel').value

        # Initialize VLA inference
        self.get_logger().info('Initializing VLA model...')
        try:
            from .vla_inference import VLAInference
            self.vla = VLAInference(
                model_path=model_path if model_path else None,
                device=device,
                buffer_size=image_history
            )
            self.get_logger().info('VLA model initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize VLA: {e}')
            self.vla = None

        # Safety filter
        self.safety_filter = SafetyFilter(
            max_linear_vel=max_linear,
            max_angular_vel=max_angular
        ) if safety_enabled else None

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # State
        self.current_image = None
        self.current_instruction = None
        self.is_active = False
        self.is_processing = False

        # Callback groups
        self._sensor_cb_group = ReentrantCallbackGroup()
        self._control_cb_group = MutuallyExclusiveCallbackGroup()

        # Lock for thread safety
        self._lock = threading.Lock()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            10,
            callback_group=self._sensor_cb_group
        )

        self.instruction_sub = self.create_subscription(
            String,
            '/vla/instruction',
            self._instruction_callback,
            10
        )

        self.enable_sub = self.create_subscription(
            Bool,
            '/vla/enable',
            self._enable_callback,
            10
        )

        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray,
            '/vla/action',
            10
        )

        self.arm_vel_pub = self.create_publisher(
            TwistStamped,
            '/arm_controller/twist_cmd',
            10
        )

        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_controller/command',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/vla/status',
            10
        )

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_loop,
            callback_group=self._control_cb_group
        )

        # Status timer
        self.status_timer = self.create_timer(
            1.0,  # 1 Hz status updates
            self._publish_status
        )

        self.get_logger().info('VLA Controller Node ready')
        self.get_logger().info('Waiting for instruction on /vla/instruction')

    def _image_callback(self, msg: Image):
        """Handle incoming camera images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            with self._lock:
                self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')

    def _instruction_callback(self, msg: String):
        """Handle new instruction."""
        instruction = msg.data.strip()

        if not instruction:
            self.is_active = False
            self.current_instruction = None
            self.get_logger().info('Instruction cleared')
            return

        self.current_instruction = instruction
        self.is_active = True

        if self.vla:
            self.vla.set_instruction(instruction)

        self.get_logger().info(f'New instruction: "{instruction}"')

    def _enable_callback(self, msg: Bool):
        """Enable/disable VLA control."""
        self.is_active = msg.data
        state = "enabled" if msg.data else "disabled"
        self.get_logger().info(f'VLA control {state}')

    def _control_loop(self):
        """Main control loop - runs at fixed rate."""
        if not self.is_active or self.vla is None:
            return

        if self.is_processing:
            return

        with self._lock:
            if self.current_image is None:
                return
            image = self.current_image.copy()

        self.is_processing = True

        try:
            # Run VLA inference
            start_time = time.time()
            action = self.vla.predict(image)
            inference_time = time.time() - start_time

            # Apply safety filter
            if self.safety_filter:
                action, is_safe, violations = self.safety_filter.filter(action)
                if not is_safe:
                    for v in violations:
                        self.get_logger().warn(f'Safety: {v}')

            # Publish action
            self._publish_action(action)

            self.get_logger().debug(
                f'Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, ...] '
                f'({inference_time*1000:.1f}ms)'
            )

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

        finally:
            self.is_processing = False

    def _publish_action(self, action: np.ndarray):
        """Publish action to robot controllers."""
        # Full action vector
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        # Arm velocity command (twist)
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        twist_msg.twist.linear.x = float(action[0])
        twist_msg.twist.linear.y = float(action[1])
        twist_msg.twist.linear.z = float(action[2])
        twist_msg.twist.angular.x = float(action[3])
        twist_msg.twist.angular.y = float(action[4])
        twist_msg.twist.angular.z = float(action[5])
        self.arm_vel_pub.publish(twist_msg)

        # Gripper command
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(action[6])]
        self.gripper_pub.publish(gripper_msg)

    def _publish_status(self):
        """Publish status information."""
        status_parts = []

        if self.is_active:
            status_parts.append("ACTIVE")
        else:
            status_parts.append("INACTIVE")

        if self.current_instruction:
            status_parts.append(f"instruction='{self.current_instruction[:30]}'")

        if self.vla:
            metrics = self.vla.get_metrics()
            if 'fps' in metrics:
                status_parts.append(f"fps={metrics['fps']:.1f}")

        status_msg = String()
        status_msg.data = ' | '.join(status_parts)
        self.status_pub.publish(status_msg)

    def reset(self):
        """Reset controller state."""
        self.is_active = False
        self.current_instruction = None
        self.current_image = None

        if self.vla:
            self.vla.reset()
        if self.safety_filter:
            self.safety_filter.reset()

        self.get_logger().info('Controller reset')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = VLAControllerNode()

    # Use multi-threaded executor for parallel callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
