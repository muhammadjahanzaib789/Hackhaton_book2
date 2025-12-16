#!/usr/bin/env python3
"""
Nav2 Navigation Client
Physical AI Book - Chapter 4: Navigation

Complete client for Nav2 navigation with goal sending,
feedback handling, and result processing.

Usage:
    ros2 run physical_ai_examples nav2_client

    # Navigate to specific pose
    ros2 run physical_ai_examples nav2_client --ros-args \
        -p goal_x:=2.0 -p goal_y:=1.0 -p goal_yaw:=1.57

Expected Output:
    [INFO] [nav2_client]: Nav2 client ready
    [INFO] [nav2_client]: Navigating to (2.0, 1.0, yaw=1.57)
    [INFO] [nav2_client]: Distance remaining: 1.5m
    [INFO] [nav2_client]: Navigation succeeded!

Dependencies:
    - rclpy
    - nav2_msgs
    - geometry_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Empty
import math
from typing import List, Tuple, Optional
from enum import Enum


class NavigationStatus(Enum):
    """Navigation status states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class Nav2Client(Node):
    """
    Full-featured Nav2 navigation client.

    Provides:
    - Single goal navigation
    - Multi-waypoint navigation
    - Initial pose setting
    - Navigation status tracking
    - Goal cancellation
    """

    def __init__(self):
        super().__init__('nav2_client')

        # Callback group for concurrent callbacks
        self._callback_group = ReentrantCallbackGroup()

        # Action clients
        self._nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self._callback_group
        )
        self._nav_through_poses_client = ActionClient(
            self, NavigateThroughPoses, 'navigate_through_poses',
            callback_group=self._callback_group
        )

        # Initial pose publisher
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10
        )

        # State tracking
        self._status = NavigationStatus.IDLE
        self._current_goal_handle = None
        self._feedback_callback = None

        # Parameters
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)

        self.get_logger().info('Nav2 client ready')

    @property
    def status(self) -> NavigationStatus:
        """Current navigation status."""
        return self._status

    def set_initial_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        frame_id: str = 'map'
    ):
        """
        Set the robot's initial pose for localization.

        Args:
            x: X position in meters
            y: Y position in meters
            yaw: Orientation in radians
            frame_id: Reference frame (default: 'map')
        """
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)

        # Set covariance (diagonal)
        msg.pose.covariance[0] = 0.25  # x
        msg.pose.covariance[7] = 0.25  # y
        msg.pose.covariance[35] = 0.07  # yaw

        self._initial_pose_pub.publish(msg)
        self.get_logger().info(f'Set initial pose: ({x}, {y}, yaw={yaw})')

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = 'map',
        feedback_callback=None
    ) -> bool:
        """
        Navigate to a single pose.

        Args:
            x: Target X position
            y: Target Y position
            yaw: Target orientation (radians)
            frame_id: Reference frame
            feedback_callback: Optional callback for progress updates

        Returns:
            True if goal was accepted
        """
        if not self._nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available')
            return False

        self._feedback_callback = feedback_callback
        self._status = NavigationStatus.NAVIGATING

        goal = NavigateToPose.Goal()
        goal.pose = self._create_pose_stamped(x, y, yaw, frame_id)

        self.get_logger().info(f'Navigating to ({x:.2f}, {y:.2f}, yaw={yaw:.2f})')

        self._send_goal_future = self._nav_to_pose_client.send_goal_async(
            goal,
            feedback_callback=self._nav_feedback_callback
        )
        self._send_goal_future.add_done_callback(self._goal_response_callback)

        return True

    def navigate_through_poses(
        self,
        waypoints: List[Tuple[float, float, float]],
        frame_id: str = 'map',
        feedback_callback=None
    ) -> bool:
        """
        Navigate through multiple waypoints.

        Args:
            waypoints: List of (x, y, yaw) tuples
            frame_id: Reference frame
            feedback_callback: Optional callback for progress

        Returns:
            True if goal was accepted
        """
        if not self._nav_through_poses_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateThroughPoses action server not available')
            return False

        self._feedback_callback = feedback_callback
        self._status = NavigationStatus.NAVIGATING

        goal = NavigateThroughPoses.Goal()
        for x, y, yaw in waypoints:
            goal.poses.append(self._create_pose_stamped(x, y, yaw, frame_id))

        self.get_logger().info(f'Navigating through {len(waypoints)} waypoints')

        self._send_goal_future = self._nav_through_poses_client.send_goal_async(
            goal,
            feedback_callback=self._nav_through_feedback_callback
        )
        self._send_goal_future.add_done_callback(self._goal_response_callback)

        return True

    def cancel_navigation(self):
        """Cancel the current navigation goal."""
        if self._current_goal_handle is not None:
            self.get_logger().info('Canceling navigation')
            self._current_goal_handle.cancel_goal_async()
            self._status = NavigationStatus.CANCELED

    def _create_pose_stamped(
        self,
        x: float,
        y: float,
        yaw: float,
        frame_id: str
    ) -> PoseStamped:
        """Create a PoseStamped message."""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        return pose

    def _goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            self._status = NavigationStatus.FAILED
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info('Navigation goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)

    def _nav_feedback_callback(self, feedback_msg):
        """Handle NavigateToPose feedback."""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose.position
        distance = feedback.distance_remaining

        self.get_logger().info(
            f'Position: ({current_pose.x:.2f}, {current_pose.y:.2f}), '
            f'Distance remaining: {distance:.2f}m'
        )

        if self._feedback_callback:
            self._feedback_callback(feedback)

    def _nav_through_feedback_callback(self, feedback_msg):
        """Handle NavigateThroughPoses feedback."""
        feedback = feedback_msg.feedback
        remaining = feedback.number_of_poses_remaining

        self.get_logger().info(f'Waypoints remaining: {remaining}')

        if self._feedback_callback:
            self._feedback_callback(feedback)

    def _get_result_callback(self, future):
        """Handle navigation result."""
        result = future.result()
        self._current_goal_handle = None

        if result.status == 4:  # SUCCEEDED
            self._status = NavigationStatus.SUCCEEDED
            self.get_logger().info('Navigation succeeded!')
        else:
            self._status = NavigationStatus.FAILED
            self.get_logger().error(f'Navigation failed with status: {result.status}')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    client = Nav2Client()

    # Get goal from parameters
    goal_x = client.get_parameter('goal_x').value
    goal_y = client.get_parameter('goal_y').value
    goal_yaw = client.get_parameter('goal_yaw').value

    # Navigate if goal is non-zero
    if goal_x != 0.0 or goal_y != 0.0:
        client.navigate_to_pose(goal_x, goal_y, goal_yaw)

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.cancel_navigation()

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
