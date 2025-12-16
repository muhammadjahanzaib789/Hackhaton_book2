#!/usr/bin/env python3
"""
Waypoint Follower Node
Physical AI Book - Chapter 4: Navigation

Implements patrol behavior by following a sequence of waypoints
in a loop or single pass.

Usage:
    ros2 run physical_ai_examples waypoint_follower

    # With custom waypoints file
    ros2 run physical_ai_examples waypoint_follower --ros-args \
        -p waypoints_file:=/path/to/waypoints.yaml

Expected Output:
    [INFO] [waypoint_follower]: Loaded 5 waypoints
    [INFO] [waypoint_follower]: Starting patrol...
    [INFO] [waypoint_follower]: Navigating to waypoint 1/5
    [INFO] [waypoint_follower]: Reached waypoint 1
    [INFO] [waypoint_follower]: Patrol complete!

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
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math
import yaml
from typing import List, Tuple, Optional
from enum import Enum


class PatrolState(Enum):
    """Patrol behavior states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETE = "complete"


class WaypointFollower(Node):
    """
    Waypoint patrol behavior.

    Features:
    - Load waypoints from YAML or parameters
    - Loop or single-pass modes
    - Wait time at each waypoint
    - Pause/resume support
    """

    def __init__(self):
        super().__init__('waypoint_follower')

        # Parameters
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('loop', False)
        self.declare_parameter('wait_duration', 2.0)

        self._loop = self.get_parameter('loop').value
        self._wait_duration = self.get_parameter('wait_duration').value

        # Load waypoints
        waypoints_file = self.get_parameter('waypoints_file').value
        if waypoints_file:
            self._waypoints = self._load_waypoints_from_file(waypoints_file)
        else:
            # Default demo waypoints
            self._waypoints = [
                (1.0, 0.0, 0.0),
                (2.0, 1.0, 1.57),
                (1.0, 2.0, 3.14),
                (0.0, 1.0, -1.57),
                (0.0, 0.0, 0.0),
            ]

        self.get_logger().info(f'Loaded {len(self._waypoints)} waypoints')

        # State
        self._state = PatrolState.IDLE
        self._current_waypoint_idx = 0
        self._goal_handle = None

        # Action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Wait timer
        self._wait_timer = None

        # Start command subscription (optional)
        self.create_subscription(
            PoseStamped, '/patrol/start',
            self._start_callback, 10
        )

    def _load_waypoints_from_file(self, filepath: str) -> List[Tuple[float, float, float]]:
        """
        Load waypoints from YAML file.

        Expected format:
        waypoints:
          - x: 1.0
            y: 0.0
            yaw: 0.0
          - x: 2.0
            y: 1.0
            yaw: 1.57
        """
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            waypoints = []
            for wp in data.get('waypoints', []):
                waypoints.append((
                    float(wp.get('x', 0.0)),
                    float(wp.get('y', 0.0)),
                    float(wp.get('yaw', 0.0))
                ))

            return waypoints
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return []

    def _start_callback(self, msg):
        """Handle start command."""
        self.start_patrol()

    def start_patrol(self):
        """Begin patrol sequence."""
        if self._state == PatrolState.NAVIGATING:
            self.get_logger().warn('Already navigating')
            return

        if not self._waypoints:
            self.get_logger().error('No waypoints loaded')
            return

        self.get_logger().info('Starting patrol...')
        self._current_waypoint_idx = 0
        self._navigate_to_current_waypoint()

    def pause_patrol(self):
        """Pause the patrol."""
        if self._state == PatrolState.NAVIGATING and self._goal_handle:
            self._goal_handle.cancel_goal_async()
            self._state = PatrolState.PAUSED
            self.get_logger().info('Patrol paused')

    def resume_patrol(self):
        """Resume paused patrol."""
        if self._state == PatrolState.PAUSED:
            self.get_logger().info('Resuming patrol')
            self._navigate_to_current_waypoint()

    def _navigate_to_current_waypoint(self):
        """Navigate to the current waypoint in the sequence."""
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return

        x, y, yaw = self._waypoints[self._current_waypoint_idx]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        self._state = PatrolState.NAVIGATING
        self.get_logger().info(
            f'Navigating to waypoint {self._current_waypoint_idx + 1}/'
            f'{len(self._waypoints)}: ({x:.2f}, {y:.2f})'
        )

        future = self._nav_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Handle goal acceptance."""
        self._goal_handle = future.result()

        if not self._goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self._state = PatrolState.IDLE
            return

        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f'Distance remaining: {feedback.distance_remaining:.2f}m'
        )

    def _result_callback(self, future):
        """Handle navigation result."""
        result = future.result()
        self._goal_handle = None

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(
                f'Reached waypoint {self._current_waypoint_idx + 1}'
            )
            self._handle_waypoint_reached()
        else:
            self.get_logger().error(f'Navigation failed: status {result.status}')
            self._state = PatrolState.IDLE

    def _handle_waypoint_reached(self):
        """Handle successful arrival at waypoint."""
        # Wait at waypoint
        if self._wait_duration > 0:
            self._state = PatrolState.WAITING
            self.get_logger().info(f'Waiting {self._wait_duration}s...')
            self._wait_timer = self.create_timer(
                self._wait_duration,
                self._wait_complete
            )
        else:
            self._advance_to_next_waypoint()

    def _wait_complete(self):
        """Handle wait timer completion."""
        if self._wait_timer:
            self._wait_timer.cancel()
            self._wait_timer = None

        self._advance_to_next_waypoint()

    def _advance_to_next_waypoint(self):
        """Move to next waypoint in sequence."""
        self._current_waypoint_idx += 1

        if self._current_waypoint_idx >= len(self._waypoints):
            if self._loop:
                self.get_logger().info('Restarting patrol loop')
                self._current_waypoint_idx = 0
                self._navigate_to_current_waypoint()
            else:
                self._state = PatrolState.COMPLETE
                self.get_logger().info('Patrol complete!')
        else:
            self._navigate_to_current_waypoint()

    @property
    def state(self) -> PatrolState:
        """Current patrol state."""
        return self._state

    @property
    def current_waypoint(self) -> int:
        """Current waypoint index."""
        return self._current_waypoint_idx

    @property
    def total_waypoints(self) -> int:
        """Total number of waypoints."""
        return len(self._waypoints)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    follower = WaypointFollower()

    # Start patrol immediately
    # In production, might wait for start command
    import time
    time.sleep(2.0)  # Wait for Nav2 to be ready
    follower.start_patrol()

    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        follower.pause_patrol()

    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
