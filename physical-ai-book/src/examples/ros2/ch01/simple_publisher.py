#!/usr/bin/env python3
"""
Simple Publisher Node
Physical AI Book - Chapter 1: ROS 2 Fundamentals

Publishes string messages to the /chatter topic at 1 Hz.

Usage:
    ros2 run physical_ai_examples simple_publisher

Expected Output:
    [INFO] [simple_publisher]: Simple Publisher has started!
    [INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 1"
    [INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 2"
    ...

Dependencies:
    - rclpy
    - std_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """
    A simple publisher node that sends string messages.

    This demonstrates:
    - Creating a ROS 2 node
    - Creating a publisher
    - Using a timer for periodic publishing
    - Logging messages
    """

    def __init__(self):
        # Initialize the node with name 'simple_publisher'
        super().__init__('simple_publisher')

        # Create a publisher
        # Parameters:
        #   - Message type: String
        #   - Topic name: 'chatter'
        #   - Queue size: 10 (buffer for outgoing messages)
        self.publisher = self.create_publisher(
            String,      # Message type
            'chatter',   # Topic name
            10           # QoS queue depth
        )

        # Create a timer that calls publish_message every 1.0 seconds
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_message)

        # Counter for message numbering
        self.count = 0

        # Log startup message
        self.get_logger().info('Simple Publisher has started!')

    def publish_message(self):
        """
        Timer callback function.
        Called every timer_period seconds to publish a message.
        """
        # Create a new String message
        msg = String()

        # Increment counter
        self.count += 1

        # Set message data
        msg.data = f'Hello, ROS 2! Count: {self.count}'

        # Publish the message
        self.publisher.publish(msg)

        # Log what we published
        self.get_logger().info(f'Publishing: "{msg.data}"')


def main(args=None):
    """
    Main entry point for the node.
    """
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create an instance of the publisher node
    node = SimplePublisher()

    try:
        # Spin the node to process callbacks
        # This blocks until the node is shut down
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
