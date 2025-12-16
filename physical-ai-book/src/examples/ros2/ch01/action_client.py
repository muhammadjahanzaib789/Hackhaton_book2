#!/usr/bin/env python3
"""
Action Client Example
Physical AI Book - Chapter 1: ROS 2 Fundamentals

Sends a goal to an action server and handles feedback/result.
Works with the standard Fibonacci action for demonstration.

Usage:
    # First, start the demo action server:
    ros2 run action_tutorials_py fibonacci_action_server

    # Then run this client:
    ros2 run physical_ai_examples action_client

Expected Output:
    [INFO] [action_client]: Waiting for action server...
    [INFO] [action_client]: Sending goal: compute Fibonacci(5)
    [INFO] [action_client]: Goal accepted!
    [INFO] [action_client]: Feedback: [0, 1, 1]
    [INFO] [action_client]: Feedback: [0, 1, 1, 2]
    [INFO] [action_client]: Feedback: [0, 1, 1, 2, 3]
    [INFO] [action_client]: Result: [0, 1, 1, 2, 3, 5]

Dependencies:
    - rclpy
    - example_interfaces

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):
    """
    An action client that requests Fibonacci sequence computation.

    This demonstrates:
    - Creating an action client
    - Sending goals
    - Handling feedback
    - Processing results
    - Handling cancellation
    """

    def __init__(self):
        super().__init__('action_client')

        # Create an action client
        # Parameters:
        #   - Node: self
        #   - Action type: Fibonacci
        #   - Action name: 'fibonacci'
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

        self.get_logger().info('Action Client initialized')

    def send_goal(self, order):
        """
        Send a goal to compute Fibonacci sequence.

        Args:
            order: The number of Fibonacci numbers to compute
        """
        # Create goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info('Waiting for action server...')

        # Wait for server to be available
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal: compute Fibonacci({order})')

        # Send goal asynchronously with feedback callback
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        # Add callback for when goal is accepted/rejected
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Called when the goal is accepted or rejected.

        Args:
            future: Future containing the goal handle
        """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return

        self.get_logger().info('Goal accepted!')

        # Request the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        Called when feedback is received from the server.

        Args:
            feedback_msg: The feedback message
        """
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        """
        Called when the action completes.

        Args:
            future: Future containing the result
        """
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

        # Shutdown after receiving result
        rclpy.shutdown()


def main(args=None):
    """
    Main entry point.
    """
    rclpy.init(args=args)

    client = FibonacciActionClient()

    # Send goal to compute Fibonacci(5)
    # Change this number to compute more/fewer terms
    client.send_goal(5)

    # Spin to process callbacks
    rclpy.spin(client)


if __name__ == '__main__':
    main()
