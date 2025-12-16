#!/usr/bin/env python3
"""
Simple Subscriber Node
Physical AI Book - Chapter 1: ROS 2 Fundamentals

Subscribes to the /chatter topic and logs received messages.

Usage:
    ros2 run physical_ai_examples simple_subscriber

Expected Output:
    [INFO] [simple_subscriber]: Simple Subscriber has started!
    [INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 1"
    [INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 2"
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


class SimpleSubscriber(Node):
    """
    A simple subscriber node that receives string messages.

    This demonstrates:
    - Creating a subscriber
    - Defining callback functions
    - Processing received messages
    """

    def __init__(self):
        # Initialize the node with name 'simple_subscriber'
        super().__init__('simple_subscriber')

        # Create a subscription
        # Parameters:
        #   - Message type: String
        #   - Topic name: 'chatter'
        #   - Callback function: self.listener_callback
        #   - Queue size: 10 (buffer for incoming messages)
        self.subscription = self.create_subscription(
            String,                    # Message type
            'chatter',                 # Topic name
            self.listener_callback,    # Callback function
            10                         # QoS queue depth
        )

        # Prevent unused variable warning
        self.subscription  # noqa: B018

        # Log startup message
        self.get_logger().info('Simple Subscriber has started!')

    def listener_callback(self, msg):
        """
        Callback function for the subscription.
        Called whenever a message is received on the topic.

        Args:
            msg: The received String message
        """
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main entry point for the node.
    """
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the subscriber node
    node = SimpleSubscriber()

    try:
        # Spin to process incoming messages
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
