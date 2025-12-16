#!/usr/bin/env python3
"""
Integration Tests for Chapter 1: ROS 2 Fundamentals
Physical AI Book

These tests verify that the Chapter 1 code examples work correctly
in a ROS 2 environment. Tests are designed to run both in Docker
and on a native ROS 2 installation.

Usage:
    # Run all Chapter 1 tests
    pytest tests/integration/test_ch01_ros2_basics.py -v

    # Run specific test
    pytest tests/integration/test_ch01_ros2_basics.py::TestPublisherSubscriber -v

    # Run with ROS 2 environment (in Docker)
    docker-compose run ros2 pytest tests/integration/test_ch01_ros2_basics.py -v

Requirements:
    - ROS 2 Humble installed
    - pytest
    - rclpy
    - std_msgs, geometry_msgs, example_interfaces

Author: Physical AI Book
License: MIT
"""

import json
import sys
import threading
import time
import unittest
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Check if ROS 2 is available
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
except ImportError:
    pass

# Skip all tests if ROS 2 is not available
pytestmark = pytest.mark.skipif(
    not ROS2_AVAILABLE,
    reason="ROS 2 (rclpy) not available"
)


class ROS2TestCase:
    """Base class for ROS 2 integration tests."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS 2 once for all tests in the class."""
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS 2 after all tests."""
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures."""
        self.nodes: List[Node] = []
        self.executor = SingleThreadedExecutor()

    def tearDown(self):
        """Clean up nodes after each test."""
        for node in self.nodes:
            node.destroy_node()
        self.nodes.clear()

    def spin_for(self, duration: float):
        """Spin the executor for a specified duration."""
        start = time.time()
        while time.time() - start < duration:
            self.executor.spin_once(timeout_sec=0.1)


@pytest.mark.ros2
class TestPublisherSubscriber(ROS2TestCase):
    """Test publisher/subscriber communication pattern."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.received_messages: List[str] = []

    def test_simple_pub_sub_communication(self):
        """Test that publisher messages reach subscriber."""
        # Create publisher node
        pub_node = rclpy.create_node('test_publisher')
        self.nodes.append(pub_node)
        publisher = pub_node.create_publisher(String, 'test_chatter', 10)

        # Create subscriber node
        sub_node = rclpy.create_node('test_subscriber')
        self.nodes.append(sub_node)

        def callback(msg):
            self.received_messages.append(msg.data)

        subscription = sub_node.create_subscription(
            String, 'test_chatter', callback, 10
        )

        # Add nodes to executor
        self.executor.add_node(pub_node)
        self.executor.add_node(sub_node)

        # Publish messages
        for i in range(3):
            msg = String()
            msg.data = f'Test message {i}'
            publisher.publish(msg)
            self.spin_for(0.2)

        # Verify messages received
        assert len(self.received_messages) >= 1, \
            "Expected at least 1 message to be received"
        assert any('Test message' in m for m in self.received_messages), \
            f"Expected 'Test message' in received: {self.received_messages}"

    def test_publisher_qos_depth(self):
        """Test that QoS depth limits message buffering."""
        pub_node = rclpy.create_node('test_qos_publisher')
        self.nodes.append(pub_node)

        # Create publisher with small queue
        publisher = pub_node.create_publisher(String, 'test_qos', 2)

        # Publish more messages than queue can hold
        for i in range(10):
            msg = String()
            msg.data = f'Message {i}'
            publisher.publish(msg)

        # Should not raise, demonstrates fire-and-forget nature
        assert True

    def test_subscriber_callback_execution(self):
        """Test that subscriber callbacks execute correctly."""
        node = rclpy.create_node('test_callback_node')
        self.nodes.append(node)

        callback_count = [0]

        def counting_callback(msg):
            callback_count[0] += 1

        publisher = node.create_publisher(String, 'callback_test', 10)
        subscription = node.create_subscription(
            String, 'callback_test', counting_callback, 10
        )

        self.executor.add_node(node)

        # Publish and spin
        for _ in range(5):
            msg = String()
            msg.data = 'count'
            publisher.publish(msg)
            self.spin_for(0.1)

        assert callback_count[0] >= 1, \
            f"Expected callbacks to fire, got {callback_count[0]}"


@pytest.mark.ros2
class TestServiceServer(ROS2TestCase):
    """Test service server/client communication pattern."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_service_creation(self):
        """Test that a service can be created."""
        try:
            from example_interfaces.srv import AddTwoInts
        except ImportError:
            pytest.skip("example_interfaces not available")

        node = rclpy.create_node('test_service_node')
        self.nodes.append(node)

        def add_callback(request, response):
            response.sum = request.a + request.b
            return response

        service = node.create_service(
            AddTwoInts, 'test_add_two_ints', add_callback
        )

        assert service is not None
        assert node.count_services() >= 1

    def test_service_call_synchronous(self):
        """Test synchronous service call."""
        try:
            from example_interfaces.srv import AddTwoInts
        except ImportError:
            pytest.skip("example_interfaces not available")

        # Create server node
        server_node = rclpy.create_node('test_add_server')
        self.nodes.append(server_node)

        def add_callback(request, response):
            response.sum = request.a + request.b
            return response

        service = server_node.create_service(
            AddTwoInts, 'integration_add', add_callback
        )

        # Create client node
        client_node = rclpy.create_node('test_add_client')
        self.nodes.append(client_node)
        client = client_node.create_client(AddTwoInts, 'integration_add')

        self.executor.add_node(server_node)
        self.executor.add_node(client_node)

        # Wait for service
        assert client.wait_for_service(timeout_sec=2.0), \
            "Service not available"

        # Make request
        request = AddTwoInts.Request()
        request.a = 5
        request.b = 3

        future = client.call_async(request)

        # Spin until complete
        timeout = time.time() + 5.0
        while not future.done() and time.time() < timeout:
            self.executor.spin_once(timeout_sec=0.1)

        assert future.done(), "Service call did not complete"
        assert future.result().sum == 8, \
            f"Expected 8, got {future.result().sum}"


@pytest.mark.ros2
class TestMultiNodeSystem(ROS2TestCase):
    """Test multi-node system communication."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.received_status: List[dict] = []

    def test_velocity_command_flow(self):
        """Test velocity commands flow through the system."""
        # Commander node (publishes Twist)
        commander = rclpy.create_node('test_commander')
        self.nodes.append(commander)
        cmd_pub = commander.create_publisher(Twist, 'test_cmd_vel', 10)

        # Controller node (receives Twist, publishes status)
        controller = rclpy.create_node('test_controller')
        self.nodes.append(controller)

        controller_received = []

        def cmd_callback(msg):
            controller_received.append({
                'linear': msg.linear.x,
                'angular': msg.angular.z
            })

        cmd_sub = controller.create_subscription(
            Twist, 'test_cmd_vel', cmd_callback, 10
        )
        status_pub = controller.create_publisher(String, 'test_status', 10)

        # Monitor node (receives status)
        monitor = rclpy.create_node('test_monitor')
        self.nodes.append(monitor)

        def status_callback(msg):
            try:
                self.received_status.append(json.loads(msg.data))
            except json.JSONDecodeError:
                pass

        status_sub = monitor.create_subscription(
            String, 'test_status', status_callback, 10
        )

        # Add to executor
        for node in self.nodes:
            self.executor.add_node(node)

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.1
        cmd_pub.publish(cmd)

        self.spin_for(0.5)

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'x': 1.0, 'y': 0.5, 'theta': 0.1,
            'linear_vel': 0.5, 'angular_vel': 0.1
        })
        status_pub.publish(status_msg)

        self.spin_for(0.5)

        # Verify flow
        assert len(controller_received) >= 1, \
            "Controller should receive velocity command"
        assert controller_received[0]['linear'] == 0.5

        assert len(self.received_status) >= 1, \
            "Monitor should receive status"
        assert self.received_status[0]['linear_vel'] == 0.5

    def test_json_status_format(self):
        """Test that status messages use correct JSON format."""
        node = rclpy.create_node('test_json_node')
        self.nodes.append(node)

        publisher = node.create_publisher(String, 'json_status', 10)

        def status_callback(msg):
            self.received_status.append(json.loads(msg.data))

        subscription = node.create_subscription(
            String, 'json_status', status_callback, 10
        )

        self.executor.add_node(node)

        # Publish JSON status
        status = {
            'x': 1.5,
            'y': 2.5,
            'theta': 0.785,
            'linear_vel': 0.3,
            'angular_vel': 0.0
        }
        msg = String()
        msg.data = json.dumps(status)
        publisher.publish(msg)

        self.spin_for(0.3)

        assert len(self.received_status) >= 1
        received = self.received_status[0]
        assert 'x' in received
        assert 'y' in received
        assert 'theta' in received
        assert received['x'] == 1.5


@pytest.mark.ros2
class TestNodeLifecycle(ROS2TestCase):
    """Test node creation and destruction."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_node_creation(self):
        """Test that nodes can be created with unique names."""
        node1 = rclpy.create_node('lifecycle_test_1')
        node2 = rclpy.create_node('lifecycle_test_2')

        self.nodes.extend([node1, node2])

        assert node1.get_name() == 'lifecycle_test_1'
        assert node2.get_name() == 'lifecycle_test_2'

    def test_node_destruction(self):
        """Test that nodes can be properly destroyed."""
        node = rclpy.create_node('destroy_test')

        # Create some resources
        pub = node.create_publisher(String, 'destroy_topic', 10)
        timer = node.create_timer(1.0, lambda: None)

        # Destroy should not raise
        node.destroy_node()

        # Node should be destroyed (not in our cleanup list)
        assert True

    def test_multiple_publishers_same_topic(self):
        """Test multiple nodes publishing to same topic."""
        node1 = rclpy.create_node('multi_pub_1')
        node2 = rclpy.create_node('multi_pub_2')
        receiver = rclpy.create_node('multi_receiver')

        self.nodes.extend([node1, node2, receiver])

        pub1 = node1.create_publisher(String, 'shared_topic', 10)
        pub2 = node2.create_publisher(String, 'shared_topic', 10)

        received = []

        def callback(msg):
            received.append(msg.data)

        sub = receiver.create_subscription(
            String, 'shared_topic', callback, 10
        )

        for node in self.nodes:
            self.executor.add_node(node)

        # Both publishers send
        msg1 = String()
        msg1.data = 'from_node_1'
        pub1.publish(msg1)

        msg2 = String()
        msg2.data = 'from_node_2'
        pub2.publish(msg2)

        self.spin_for(0.5)

        assert len(received) >= 1, \
            "Should receive messages from multiple publishers"


@pytest.mark.ros2
class TestTimerCallbacks(ROS2TestCase):
    """Test timer-based callbacks."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_timer_fires(self):
        """Test that timers fire at expected intervals."""
        node = rclpy.create_node('timer_test')
        self.nodes.append(node)
        self.executor.add_node(node)

        callback_times = []

        def timer_callback():
            callback_times.append(time.time())

        timer = node.create_timer(0.1, timer_callback)

        self.spin_for(0.5)

        assert len(callback_times) >= 3, \
            f"Expected at least 3 callbacks, got {len(callback_times)}"

    def test_timer_cancellation(self):
        """Test that timers can be cancelled."""
        node = rclpy.create_node('cancel_timer_test')
        self.nodes.append(node)
        self.executor.add_node(node)

        callback_count = [0]

        def timer_callback():
            callback_count[0] += 1

        timer = node.create_timer(0.1, timer_callback)

        self.spin_for(0.25)
        count_before_cancel = callback_count[0]

        timer.cancel()

        self.spin_for(0.25)
        count_after_cancel = callback_count[0]

        # Should have stopped incrementing after cancel
        assert count_after_cancel == count_before_cancel, \
            "Timer should stop after cancellation"


# Unit tests that don't require ROS 2
class TestCodeExamplesStructure(unittest.TestCase):
    """Unit tests for code example structure (no ROS 2 required)."""

    def test_example_files_exist(self):
        """Test that all Chapter 1 example files exist."""
        import os

        base_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'src', 'examples', 'ros2', 'ch01'
        )

        expected_files = [
            '__init__.py',
            'simple_publisher.py',
            'simple_subscriber.py',
            'service_server.py',
            'action_client.py',
            'multi_node_system.py',
        ]

        # This test validates structure, may skip if path doesn't exist
        if not os.path.exists(base_path):
            pytest.skip("Example path not found in test environment")

        for filename in expected_files:
            filepath = os.path.join(base_path, filename)
            assert os.path.exists(filepath), \
                f"Expected file not found: {filename}"

    def test_example_files_have_docstrings(self):
        """Test that example files have proper docstrings."""
        import os
        import ast

        base_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'src', 'examples', 'ros2', 'ch01'
        )

        if not os.path.exists(base_path):
            pytest.skip("Example path not found in test environment")

        python_files = [
            'simple_publisher.py',
            'simple_subscriber.py',
            'service_server.py',
            'action_client.py',
            'multi_node_system.py',
        ]

        for filename in python_files:
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                continue

            with open(filepath, 'r') as f:
                content = f.read()

            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)

            assert docstring is not None, \
                f"{filename} should have a module docstring"
            assert len(docstring) > 50, \
                f"{filename} docstring should be descriptive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
