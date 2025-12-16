#!/usr/bin/env python3
"""
Multi-Node Communication System
Physical AI Book - Chapter 1: ROS 2 Fundamentals

Demonstrates a complete system with multiple nodes communicating
via topics, services, and parameter handling.

This file contains three nodes that can be run together or separately:
1. Commander - Sends velocity commands
2. Controller - Processes commands and maintains state
3. Monitor - Displays system status

Usage (run in separate terminals):
    # Terminal 1: Controller
    ros2 run physical_ai_examples multi_node_system --ros-args -p node:=controller

    # Terminal 2: Commander
    ros2 run physical_ai_examples multi_node_system --ros-args -p node:=commander

    # Terminal 3: Monitor
    ros2 run physical_ai_examples multi_node_system --ros-args -p node:=monitor

    # Or run all at once (requires executor):
    ros2 run physical_ai_examples multi_node_system --ros-args -p node:=all

Expected Output:
    Controller: Processing commands, publishing status
    Commander: Sending velocity commands at 1 Hz
    Monitor: Displaying status updates

Dependencies:
    - rclpy
    - std_msgs
    - geometry_msgs

Author: Physical AI Book
License: MIT
"""

import json
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class Commander(Node):
    """
    Commander node that publishes velocity commands.

    Publishes:
        /cmd_vel (Twist): Velocity commands

    This simulates a higher-level planner sending movement commands.
    """

    def __init__(self):
        super().__init__('commander')

        # Create publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for publishing at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_command)

        # Command sequence
        self.commands = [
            (0.5, 0.0),   # Forward
            (0.5, 0.0),   # Forward
            (0.0, 0.5),   # Turn left
            (0.5, 0.0),   # Forward
            (0.0, -0.5),  # Turn right
            (0.5, 0.0),   # Forward
            (0.0, 0.0),   # Stop
        ]
        self.cmd_index = 0

        self.get_logger().info('Commander started - sending velocity commands')

    def publish_command(self):
        """Publish the next velocity command."""
        if self.cmd_index >= len(self.commands):
            self.cmd_index = 0  # Loop commands

        linear, angular = self.commands[self.cmd_index]

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        self.cmd_pub.publish(msg)

        self.get_logger().info(
            f'Command: linear={linear:.1f} m/s, angular={angular:.1f} rad/s'
        )

        self.cmd_index += 1


class Controller(Node):
    """
    Controller node that processes commands and maintains robot state.

    Subscribes:
        /cmd_vel (Twist): Velocity commands

    Publishes:
        /status (String): JSON-encoded robot status

    This simulates a low-level controller that executes commands
    and tracks the robot's state.
    """

    def __init__(self):
        super().__init__('controller')

        # Subscribe to velocity commands
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10
        )

        # Publish robot status
        self.status_pub = self.create_publisher(String, 'status', 10)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Heading in radians
        self.current_cmd = Twist()

        # Timer for publishing status at 2 Hz
        self.timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info('Controller started - processing commands')

    def cmd_callback(self, msg):
        """Process received velocity command."""
        self.current_cmd = msg

        # Simple kinematic update (Euler integration)
        dt = 0.5  # Assume 2 Hz update rate
        self.theta += msg.angular.z * dt
        self.x += msg.linear.x * dt * 0.5  # Simplified
        self.y += msg.linear.x * dt * 0.3  # Simplified

        self.get_logger().debug(
            f'Received cmd: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}'
        )

    def publish_status(self):
        """Publish current robot status as JSON."""
        status = {
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'theta': round(self.theta, 2),
            'linear_vel': round(self.current_cmd.linear.x, 2),
            'angular_vel': round(self.current_cmd.angular.z, 2),
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


class Monitor(Node):
    """
    Monitor node that displays system status.

    Subscribes:
        /status (String): JSON-encoded robot status

    This simulates a monitoring/visualization component.
    """

    def __init__(self):
        super().__init__('monitor')

        # Subscribe to status
        self.status_sub = self.create_subscription(
            String,
            'status',
            self.status_callback,
            10
        )

        self.get_logger().info('Monitor started - displaying status')

    def status_callback(self, msg):
        """Display received status."""
        try:
            status = json.loads(msg.data)
            self.get_logger().info(
                f'Robot Status: '
                f'pos=({status["x"]:.2f}, {status["y"]:.2f}), '
                f'theta={status["theta"]:.2f} rad, '
                f'vel=({status["linear_vel"]:.2f}, {status["angular_vel"]:.2f})'
            )
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid status JSON: {msg.data}')


def main(args=None):
    """
    Main entry point.

    Supports running individual nodes or all nodes together.
    """
    rclpy.init(args=args)

    # Create a temporary node to get parameters
    temp_node = rclpy.create_node('_temp')
    temp_node.declare_parameter('node', 'all')
    node_type = temp_node.get_parameter('node').value
    temp_node.destroy_node()

    nodes = []

    if node_type == 'commander':
        nodes.append(Commander())
    elif node_type == 'controller':
        nodes.append(Controller())
    elif node_type == 'monitor':
        nodes.append(Monitor())
    elif node_type == 'all':
        # Run all nodes together
        nodes = [Controller(), Commander(), Monitor()]
    else:
        print(f'Unknown node type: {node_type}')
        print('Use: commander, controller, monitor, or all')
        return

    try:
        if len(nodes) == 1:
            # Single node mode
            rclpy.spin(nodes[0])
        else:
            # Multi-node mode with executor
            executor = MultiThreadedExecutor()
            for node in nodes:
                executor.add_node(node)
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
