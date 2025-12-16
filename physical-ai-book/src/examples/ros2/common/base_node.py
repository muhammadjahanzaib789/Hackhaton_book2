#!/usr/bin/env python3
"""
Base Node Class for Physical AI Book Examples
Provides common patterns and utilities for ROS 2 nodes.

This base class includes:
- Automatic parameter declaration
- Lifecycle logging
- Error handling patterns
- Common utility methods

Usage:
    from physical_ai_examples.common import BasePhysicalAINode

    class MyNode(BasePhysicalAINode):
        def __init__(self):
            super().__init__('my_node')
            # Custom initialization

Author: Physical AI Book
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import traceback

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class BasePhysicalAINode(Node, ABC):
    """
    Base class for Physical AI Book example nodes.

    Provides:
    - Standardized initialization
    - Common QoS profiles
    - Error handling wrappers
    - Lifecycle logging
    """

    # Common QoS profiles
    QOS_RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    QOS_BEST_EFFORT = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )

    QOS_SENSOR = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5
    )

    def __init__(
        self,
        node_name: str,
        *,
        namespace: str = '',
        use_sim_time: bool = True,
        enable_rosout: bool = True,
    ):
        """
        Initialize the base node.

        Args:
            node_name: Name of the ROS 2 node
            namespace: Optional namespace prefix
            use_sim_time: Whether to use simulation time (default True)
            enable_rosout: Whether to enable rosout logging
        """
        super().__init__(
            node_name,
            namespace=namespace,
            enable_rosout=enable_rosout,
        )

        # Declare common parameters
        self.declare_parameter('use_sim_time', use_sim_time)

        # Callback groups for concurrent execution
        self._reentrant_group = ReentrantCallbackGroup()
        self._exclusive_group = MutuallyExclusiveCallbackGroup()

        # Track publishers and subscribers for cleanup
        self._tracked_publishers: List = []
        self._tracked_subscriptions: List = []
        self._tracked_timers: List = []

        self.get_logger().info(f'{node_name} initialized')

    @property
    def reentrant_group(self) -> ReentrantCallbackGroup:
        """Get the reentrant callback group for concurrent callbacks."""
        return self._reentrant_group

    @property
    def exclusive_group(self) -> MutuallyExclusiveCallbackGroup:
        """Get the exclusive callback group for sequential callbacks."""
        return self._exclusive_group

    def safe_callback(self, callback):
        """
        Wrap a callback with error handling.

        Args:
            callback: The callback function to wrap

        Returns:
            Wrapped callback that catches and logs exceptions
        """
        def wrapper(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                self.get_logger().error(
                    f'Exception in {callback.__name__}: {e}\n'
                    f'{traceback.format_exc()}'
                )
        return wrapper

    def create_tracked_publisher(
        self,
        msg_type: Type,
        topic: str,
        qos: QoSProfile = None,
        **kwargs
    ):
        """
        Create a publisher and track it for cleanup.

        Args:
            msg_type: Message type class
            topic: Topic name
            qos: QoS profile (defaults to reliable)
            **kwargs: Additional arguments for create_publisher

        Returns:
            The created publisher
        """
        qos = qos or self.QOS_RELIABLE
        pub = self.create_publisher(msg_type, topic, qos, **kwargs)
        self._tracked_publishers.append(pub)
        return pub

    def create_tracked_subscription(
        self,
        msg_type: Type,
        topic: str,
        callback,
        qos: QoSProfile = None,
        **kwargs
    ):
        """
        Create a subscription and track it for cleanup.

        Args:
            msg_type: Message type class
            topic: Topic name
            callback: Callback function
            qos: QoS profile (defaults to reliable)
            **kwargs: Additional arguments for create_subscription

        Returns:
            The created subscription
        """
        qos = qos or self.QOS_RELIABLE
        sub = self.create_subscription(
            msg_type,
            topic,
            self.safe_callback(callback),
            qos,
            **kwargs
        )
        self._tracked_subscriptions.append(sub)
        return sub

    def create_tracked_timer(
        self,
        period: float,
        callback,
        callback_group=None
    ):
        """
        Create a timer and track it for cleanup.

        Args:
            period: Timer period in seconds
            callback: Callback function
            callback_group: Optional callback group

        Returns:
            The created timer
        """
        timer = self.create_timer(
            period,
            self.safe_callback(callback),
            callback_group=callback_group or self._exclusive_group
        )
        self._tracked_timers.append(timer)
        return timer

    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value with a default.

        Args:
            name: Parameter name
            default: Default value if not set

        Returns:
            Parameter value or default
        """
        try:
            return self.get_parameter(name).value
        except rclpy.exceptions.ParameterUninitializedException:
            return default

    def log_once(self, message: str, level: str = 'info'):
        """
        Log a message only once (useful in callbacks).

        Args:
            message: Message to log
            level: Log level ('debug', 'info', 'warn', 'error')
        """
        if not hasattr(self, '_logged_messages'):
            self._logged_messages = set()

        if message not in self._logged_messages:
            self._logged_messages.add(message)
            logger = self.get_logger()
            getattr(logger, level)(message)

    def destroy_node(self):
        """Clean up resources before destruction."""
        self.get_logger().info(f'{self.get_name()} shutting down')

        # Cancel all timers
        for timer in self._tracked_timers:
            timer.cancel()

        super().destroy_node()
