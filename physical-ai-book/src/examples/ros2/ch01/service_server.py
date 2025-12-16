#!/usr/bin/env python3
"""
Service Server Example
Physical AI Book - Chapter 1: ROS 2 Fundamentals

Provides an 'add_two_ints' service that adds two integers.

Usage:
    ros2 run physical_ai_examples service_server

Test with:
    ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 3}"

Expected Output:
    Server:
        [INFO] [add_service]: Add Service ready!
        [INFO] [add_service]: Request: 5 + 3 = 8

    Client:
        sum: 8

Dependencies:
    - rclpy
    - example_interfaces

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddService(Node):
    """
    A service server that adds two integers.

    This demonstrates:
    - Creating a service server
    - Handling service requests
    - Returning service responses
    """

    def __init__(self):
        # Initialize the node
        super().__init__('add_service')

        # Create a service
        # Parameters:
        #   - Service type: AddTwoInts
        #   - Service name: 'add_two_ints'
        #   - Callback function: self.add_callback
        self.srv = self.create_service(
            AddTwoInts,           # Service type
            'add_two_ints',       # Service name
            self.add_callback     # Callback function
        )

        # Log that service is ready
        self.get_logger().info('Add Service ready!')

    def add_callback(self, request, response):
        """
        Service callback function.
        Called when a client sends a request.

        Args:
            request: The service request containing 'a' and 'b'
            response: The service response to fill in

        Returns:
            The filled response object
        """
        # Perform the addition
        response.sum = request.a + request.b

        # Log the request
        self.get_logger().info(
            f'Request: {request.a} + {request.b} = {response.sum}'
        )

        # Return the response
        return response


def main(args=None):
    """
    Main entry point.
    """
    rclpy.init(args=args)

    node = AddService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
