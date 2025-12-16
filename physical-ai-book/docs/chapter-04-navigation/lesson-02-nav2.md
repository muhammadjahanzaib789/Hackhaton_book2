---
sidebar_position: 2
title: "Lesson 2: Nav2 Navigation Stack"
description: "Autonomous navigation with the ROS 2 Navigation Stack"
---

# Nav2 Navigation Stack

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand Nav2 architecture and components
2. Configure Nav2 for humanoid navigation
3. Send navigation goals programmatically
4. Handle navigation feedback and results

## Prerequisites

- Completed Lesson 1 (SLAM Fundamentals)
- Map available for the environment
- Understanding of ROS 2 actions

## What is Nav2?

Nav2 is the ROS 2 Navigation Stack - a complete solution for autonomous robot navigation.

```
┌─────────────────────────────────────────────────────────────┐
│                    Nav2 Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  BT Navigator                        │   │
│  │        (Behavior Tree-based task execution)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│            ┌─────────────┼─────────────┐                   │
│            ▼             ▼             ▼                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │   Planner    │ │  Controller  │ │  Recovery    │       │
│  │   Server     │ │   Server     │ │  Server      │       │
│  │ (Global Plan)│ │ (Local Plan) │ │ (Stuck Help) │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
│            │             │             │                   │
│            ▼             ▼             ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Costmap 2D (Global + Local)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│            ┌─────────────┼─────────────┐                   │
│            ▼             ▼             ▼                   │
│       Map Server    AMCL/SLAM      Sensors                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **BT Navigator** | Orchestrates navigation | NavigateToPose action |
| **Planner Server** | Global path planning | NavFn, Smac Planners |
| **Controller Server** | Local trajectory following | DWB, TEB, MPPI |
| **Recovery Server** | Handles stuck situations | Spin, backup, wait |
| **Costmap 2D** | Obstacle representation | Global and local maps |
| **Map Server** | Provides static map | Occupancy grid |
| **AMCL** | Robot localization | Particle filter |

## Installing Nav2

```bash
# Install Nav2
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Verify installation
ros2 pkg list | grep nav2
```

## Nav2 Configuration

### Main Parameters File

Create `nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["FollowPath"]

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    general_goal_checker:
      stateful: True
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25

    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: True
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: True
      stateful: True
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
      BaseObstacle.scale: 0.02
      PathAlign.scale: 32.0
      PathAlign.forward_point_distance: 0.1
      GoalAlign.scale: 24.0
      GoalAlign.forward_point_distance: 0.1
      PathDist.scale: 32.0
      GoalDist.scale: 24.0
      RotateToGoal.scale: 32.0
      RotateToGoal.slowing_factor: 5.0
      RotateToGoal.lookahead_time: -1.0

planner_server:
  ros__parameters:
    use_sim_time: True
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.3
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
```

## Launching Nav2

```python
#!/usr/bin/env python3
"""
Nav2 Launch File
Physical AI Book - Chapter 4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_file = LaunchConfiguration('map')

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    params_file = os.path.join(
        get_package_share_directory('physical_ai_book'),
        'config', 'nav2_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', description='Path to map yaml file'),

        # Include Nav2 bringup
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'map': map_file,
                'params_file': params_file,
            }.items(),
        ),
    ])
```

## Sending Navigation Goals

### Using RViz

1. Launch Nav2 with RViz
2. Click "2D Pose Estimate" to set initial pose
3. Click "Nav2 Goal" to send navigation goal

### Programmatic Navigation

```python
#!/usr/bin/env python3
"""
Navigation Goal Sender
Physical AI Book - Chapter 4

Send navigation goals to Nav2 using the NavigateToPose action.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math


class NavigationClient(Node):
    """Client for sending navigation goals."""

    def __init__(self):
        super().__init__('navigation_client')

        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        self.get_logger().info('Navigation client ready')

    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        """
        Send a navigation goal.

        Args:
            x: Target x position (meters)
            y: Target y position (meters)
            yaw: Target orientation (radians)
        """
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Set orientation (quaternion from yaw)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2)

        self.get_logger().info(f'Sending goal: ({x}, {y}, {yaw})')

        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return

        self.get_logger().info('Goal accepted!')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose

        self.get_logger().info(
            f'Current position: ({current_pose.position.x:.2f}, '
            f'{current_pose.position.y:.2f})'
        )

    def get_result_callback(self, future):
        """Handle navigation result."""
        result = future.result().result

        if result:
            self.get_logger().info('Navigation succeeded!')
        else:
            self.get_logger().error('Navigation failed!')


def main(args=None):
    rclpy.init(args=args)

    client = NavigationClient()

    # Send goal to position (2, 1) facing east
    client.send_goal(2.0, 1.0, 0.0)

    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Waypoint Following

```python
#!/usr/bin/env python3
"""
Waypoint Follower
Physical AI Book - Chapter 4

Navigate through a sequence of waypoints.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateThroughPoses
from geometry_msgs.msg import PoseStamped
import math
from typing import List, Tuple


class WaypointFollower(Node):
    """Navigate through multiple waypoints."""

    def __init__(self):
        super().__init__('waypoint_follower')

        self._action_client = ActionClient(
            self, NavigateThroughPoses, 'navigate_through_poses'
        )

        self.get_logger().info('Waypoint follower ready')

    def follow_waypoints(self, waypoints: List[Tuple[float, float, float]]):
        """
        Navigate through a list of waypoints.

        Args:
            waypoints: List of (x, y, yaw) tuples
        """
        self._action_client.wait_for_server()

        goal_msg = NavigateThroughPoses.Goal()

        for x, y, yaw in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
            goal_msg.poses.append(pose)

        self.get_logger().info(f'Following {len(waypoints)} waypoints')

        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Waypoint {feedback.number_of_poses_remaining} remaining'
        )

    def goal_response_callback(self, future):
        """Handle goal response."""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Waypoint navigation started')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """Handle result."""
        self.get_logger().info('Waypoint navigation complete!')


def main(args=None):
    rclpy.init(args=args)

    follower = WaypointFollower()

    # Define waypoints (x, y, yaw)
    waypoints = [
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 1.57),
        (1.0, 2.0, 3.14),
        (0.0, 1.0, -1.57),
        (0.0, 0.0, 0.0),
    ]

    follower.follow_waypoints(waypoints)

    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **Nav2** provides complete autonomous navigation
2. **Costmaps** represent obstacles for planning
3. **Planners** compute global and local paths
4. **Actions** enable asynchronous goal handling
5. **Waypoints** allow multi-point navigation

## Next Steps

In the [next lesson](./lesson-03-path-planning.md), we will:
- Dive into path planning algorithms
- Compare different planner options
- Customize planning behavior

## Additional Resources

- [Nav2 Documentation](https://navigation.ros.org/)
- [Behavior Trees](https://www.behaviortree.dev/)
- [Costmap2D](https://navigation.ros.org/configuration/packages/costmap-plugins/index.html)
