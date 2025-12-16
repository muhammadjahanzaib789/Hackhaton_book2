---
sidebar_position: 3
title: "Lesson 3: Grasping and Manipulation"
description: "Implementing pick and place operations for humanoid robots"
---

# Grasping and Manipulation

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand grasp planning fundamentals
2. Control parallel grippers programmatically
3. Implement pick and place operations
4. Handle grasp failures gracefully

## Prerequisites

- Completed Lessons 1-2 of this chapter
- MoveIt2 configured for your robot
- Understanding of pose transformations

## Grasping Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Pick and Place Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Object Detection                                        │
│     └─▶ Find object pose in scene                          │
│                                                             │
│  2. Grasp Planning                                          │
│     └─▶ Compute valid grasp poses                          │
│                                                             │
│  3. Approach                                                │
│     └─▶ Move to pre-grasp position                         │
│                                                             │
│  4. Grasp                                                   │
│     └─▶ Close gripper on object                            │
│                                                             │
│  5. Lift                                                    │
│     └─▶ Lift object from surface                           │
│                                                             │
│  6. Transport                                               │
│     └─▶ Move to place location                             │
│                                                             │
│  7. Place                                                   │
│     └─▶ Lower and release object                           │
│                                                             │
│  8. Retreat                                                 │
│     └─▶ Move away from placed object                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Gripper Control

### Parallel Gripper Interface

```python
#!/usr/bin/env python3
"""
Parallel Gripper Controller
Physical AI Book - Chapter 5

Controls a parallel jaw gripper via ROS 2.

Usage:
    ros2 run physical_ai_examples gripper_controller

Expected Output:
    [INFO] [gripper_controller]: Gripper ready
    [INFO] [gripper_controller]: Opening gripper...
    [INFO] [gripper_controller]: Gripper opened
    [INFO] [gripper_controller]: Closing gripper...
    [INFO] [gripper_controller]: Gripper closed

Dependencies:
    - rclpy
    - control_msgs
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import Optional
import time


class GripperController(Node):
    """
    Parallel gripper controller.

    Supports both action-based and direct joint control.
    """

    def __init__(self):
        super().__init__('gripper_controller')

        # Parameters
        self.declare_parameter('max_width', 0.08)  # 8cm max opening
        self.declare_parameter('max_effort', 50.0)  # 50N max force
        self.declare_parameter('use_action', True)

        self._max_width = self.get_parameter('max_width').value
        self._max_effort = self.get_parameter('max_effort').value
        self._use_action = self.get_parameter('use_action').value

        # Current state
        self._current_position = 0.0
        self._is_gripping = False

        if self._use_action:
            # Action client for gripper
            self._gripper_client = ActionClient(
                self, GripperCommand, '/gripper_controller/gripper_cmd'
            )
        else:
            # Direct joint position publisher
            self._joint_pub = self.create_publisher(
                Float64MultiArray, '/gripper_controller/commands', 10
            )

        # Joint state subscription
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_callback, 10
        )

        self.get_logger().info('Gripper ready')

    def _joint_state_callback(self, msg):
        """Update gripper position from joint states."""
        gripper_joints = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        for i, name in enumerate(msg.name):
            if name in gripper_joints:
                self._current_position = abs(msg.position[i]) * 2  # Total width

    def open(self, width: Optional[float] = None) -> bool:
        """
        Open the gripper.

        Args:
            width: Target opening width (default: max width)

        Returns:
            True if command succeeded
        """
        if width is None:
            width = self._max_width

        self.get_logger().info(f'Opening gripper to {width*100:.1f}cm')
        return self._set_position(width, effort=self._max_effort)

    def close(self, effort: Optional[float] = None) -> bool:
        """
        Close the gripper.

        Args:
            effort: Gripping force (default: max effort)

        Returns:
            True if command succeeded
        """
        if effort is None:
            effort = self._max_effort

        self.get_logger().info(f'Closing gripper with {effort:.1f}N')
        return self._set_position(0.0, effort=effort)

    def grasp(self, width: float = 0.0, effort: Optional[float] = None) -> bool:
        """
        Grasp an object.

        Args:
            width: Minimum grasp width (object size)
            effort: Gripping force

        Returns:
            True if grasp likely succeeded
        """
        if effort is None:
            effort = self._max_effort * 0.8  # 80% of max

        result = self._set_position(width, effort=effort)

        # Check if gripper stopped (object grasped)
        time.sleep(0.5)
        self._is_gripping = self._current_position > width * 0.5

        return self._is_gripping

    def _set_position(self, position: float, effort: float) -> bool:
        """Set gripper position."""
        if self._use_action:
            return self._send_action(position, effort)
        else:
            return self._send_command(position)

    def _send_action(self, position: float, effort: float) -> bool:
        """Send gripper command via action."""
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position / 2  # Per-finger position
        goal.command.max_effort = effort

        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)

        return result_future.result().result.reached_goal

    def _send_command(self, position: float) -> bool:
        """Send direct joint command."""
        msg = Float64MultiArray()
        msg.data = [position / 2, position / 2]  # Symmetric
        self._joint_pub.publish(msg)
        return True

    @property
    def is_gripping(self) -> bool:
        """Check if gripper is currently gripping."""
        return self._is_gripping

    @property
    def current_width(self) -> float:
        """Current gripper opening width."""
        return self._current_position
```

## Grasp Planning

### Grasp Pose Generation

```python
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from typing import List, Tuple


class GraspPlanner:
    """
    Simple grasp planner for rectangular objects.

    Generates grasp poses based on object geometry.
    """

    def __init__(self, approach_distance: float = 0.1,
                 lift_height: float = 0.05):
        """
        Args:
            approach_distance: Pre-grasp offset from object
            lift_height: Height to lift after grasping
        """
        self.approach_distance = approach_distance
        self.lift_height = lift_height

    def generate_grasps(self, object_pose: Pose,
                        object_dimensions: Tuple[float, float, float],
                        num_grasps: int = 4) -> List[dict]:
        """
        Generate grasp poses for a box-shaped object.

        Args:
            object_pose: Object pose (center)
            object_dimensions: (length, width, height)
            num_grasps: Number of grasp candidates

        Returns:
            List of grasp dictionaries with poses
        """
        grasps = []
        length, width, height = object_dimensions

        # Top-down grasps along different axes
        grasp_orientations = [
            (0, 0, 0),           # Grasp along X
            (0, 0, np.pi/2),    # Grasp along Y
            (0, 0, np.pi),      # Grasp along -X
            (0, 0, -np.pi/2),   # Grasp along -Y
        ]

        for i, (roll, pitch, yaw) in enumerate(grasp_orientations[:num_grasps]):
            grasp = {
                'grasp_pose': self._create_grasp_pose(
                    object_pose, height, roll, pitch, yaw
                ),
                'pre_grasp_pose': self._create_approach_pose(
                    object_pose, height, roll, pitch, yaw
                ),
                'post_grasp_pose': self._create_lift_pose(
                    object_pose, height, roll, pitch, yaw
                ),
                'grasp_width': width if i % 2 == 0 else length,
                'score': 1.0 - 0.1 * i,  # Prefer first grasps
            }
            grasps.append(grasp)

        return grasps

    def _create_grasp_pose(self, object_pose: Pose, height: float,
                           roll: float, pitch: float, yaw: float) -> Pose:
        """Create the actual grasp pose."""
        grasp = Pose()

        # Position: at object center, slightly above
        grasp.position.x = object_pose.position.x
        grasp.position.y = object_pose.position.y
        grasp.position.z = object_pose.position.z + height / 2

        # Orientation: pointing down with specified rotation
        q = self._euler_to_quaternion(roll, pitch + np.pi, yaw)
        grasp.orientation.x = q[0]
        grasp.orientation.y = q[1]
        grasp.orientation.z = q[2]
        grasp.orientation.w = q[3]

        return grasp

    def _create_approach_pose(self, object_pose: Pose, height: float,
                              roll: float, pitch: float, yaw: float) -> Pose:
        """Create pre-grasp approach pose."""
        approach = self._create_grasp_pose(object_pose, height, roll, pitch, yaw)
        approach.position.z += self.approach_distance
        return approach

    def _create_lift_pose(self, object_pose: Pose, height: float,
                          roll: float, pitch: float, yaw: float) -> Pose:
        """Create post-grasp lift pose."""
        lift = self._create_grasp_pose(object_pose, height, roll, pitch, yaw)
        lift.position.z += self.lift_height
        return lift

    def _euler_to_quaternion(self, roll: float, pitch: float,
                             yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        )
```

## Pick and Place Action Server

```python
#!/usr/bin/env python3
"""
Pick and Place Action Server
Physical AI Book - Chapter 5

ROS 2 action server for pick and place operations.

Usage:
    ros2 run physical_ai_examples pick_place_server

Expected Output:
    [INFO] [pick_place_server]: Pick and place server ready
    [INFO] [pick_place_server]: Received pick request for object_1
    [INFO] [pick_place_server]: Planning approach...
    [INFO] [pick_place_server]: Executing pick...
    [INFO] [pick_place_server]: Pick succeeded

Dependencies:
    - rclpy
    - moveit_py
    - control_msgs
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from typing import Optional
import time


# Custom action definition would normally be in a separate package
# For this example, we'll use a simplified approach


class PickPlaceServer(Node):
    """
    Pick and place action server.

    Coordinates motion planning and gripper control
    for manipulation tasks.
    """

    def __init__(self):
        super().__init__('pick_place_server')

        self._callback_group = ReentrantCallbackGroup()

        # Initialize components (would connect to MoveIt and gripper)
        self._grasp_planner = GraspPlanner()
        self._gripper = GripperController()

        # Planning scene objects
        self._objects = {}

        self.get_logger().info('Pick and place server ready')

    def pick(self, object_name: str, object_pose: Pose,
             object_dimensions: tuple) -> bool:
        """
        Execute a pick operation.

        Args:
            object_name: Name of object to pick
            object_pose: Object pose in world frame
            object_dimensions: (length, width, height)

        Returns:
            True if pick succeeded
        """
        self.get_logger().info(f'Picking {object_name}')

        # Generate grasp candidates
        grasps = self._grasp_planner.generate_grasps(
            object_pose, object_dimensions
        )

        if not grasps:
            self.get_logger().error('No valid grasps found')
            return False

        # Try each grasp until one succeeds
        for i, grasp in enumerate(grasps):
            self.get_logger().info(f'Trying grasp {i+1}/{len(grasps)}')

            success = self._execute_pick(
                grasp['pre_grasp_pose'],
                grasp['grasp_pose'],
                grasp['post_grasp_pose'],
                grasp['grasp_width']
            )

            if success:
                self._objects[object_name] = 'attached'
                return True

        self.get_logger().error('All grasp attempts failed')
        return False

    def place(self, object_name: str, place_pose: Pose) -> bool:
        """
        Execute a place operation.

        Args:
            object_name: Name of object to place
            place_pose: Target placement pose

        Returns:
            True if place succeeded
        """
        if object_name not in self._objects:
            self.get_logger().error(f'Object {object_name} not in gripper')
            return False

        self.get_logger().info(f'Placing {object_name}')

        # Create place poses
        pre_place = Pose()
        pre_place.position = place_pose.position
        pre_place.position.z += 0.1  # Approach from above
        pre_place.orientation = place_pose.orientation

        post_place = Pose()
        post_place.position = place_pose.position
        post_place.position.z += 0.1  # Retreat upward
        post_place.orientation = place_pose.orientation

        success = self._execute_place(pre_place, place_pose, post_place)

        if success:
            del self._objects[object_name]

        return success

    def _execute_pick(self, pre_grasp: Pose, grasp: Pose,
                      post_grasp: Pose, grasp_width: float) -> bool:
        """Execute the pick sequence."""
        try:
            # 1. Open gripper
            self.get_logger().info('Opening gripper')
            if not self._gripper.open():
                return False

            # 2. Move to pre-grasp
            self.get_logger().info('Moving to approach pose')
            if not self._plan_and_execute(pre_grasp):
                return False

            # 3. Move to grasp pose (Cartesian)
            self.get_logger().info('Approaching object')
            if not self._plan_cartesian([grasp]):
                return False

            # 4. Close gripper
            self.get_logger().info('Grasping')
            if not self._gripper.grasp(width=grasp_width * 0.8):
                self.get_logger().warn('Grasp may have failed')

            # 5. Attach object to gripper (collision avoidance)
            self._attach_object_to_gripper()

            # 6. Lift
            self.get_logger().info('Lifting')
            if not self._plan_cartesian([post_grasp]):
                return False

            return True

        except Exception as e:
            self.get_logger().error(f'Pick failed: {e}')
            return False

    def _execute_place(self, pre_place: Pose, place: Pose,
                       post_place: Pose) -> bool:
        """Execute the place sequence."""
        try:
            # 1. Move to pre-place
            self.get_logger().info('Moving to place approach')
            if not self._plan_and_execute(pre_place):
                return False

            # 2. Lower to place pose
            self.get_logger().info('Lowering object')
            if not self._plan_cartesian([place]):
                return False

            # 3. Open gripper
            self.get_logger().info('Releasing')
            if not self._gripper.open():
                return False

            # 4. Detach object
            self._detach_object_from_gripper()

            # 5. Retreat
            self.get_logger().info('Retreating')
            if not self._plan_cartesian([post_place]):
                return False

            return True

        except Exception as e:
            self.get_logger().error(f'Place failed: {e}')
            return False

    def _plan_and_execute(self, target: Pose) -> bool:
        """Plan and execute motion to target pose."""
        # In real implementation, would use MoveIt
        self.get_logger().debug(f'Planning to {target.position}')
        time.sleep(0.5)  # Simulate planning
        return True

    def _plan_cartesian(self, waypoints: list) -> bool:
        """Plan and execute Cartesian path."""
        self.get_logger().debug(f'Cartesian path: {len(waypoints)} waypoints')
        time.sleep(0.3)  # Simulate motion
        return True

    def _attach_object_to_gripper(self):
        """Attach object to gripper in planning scene."""
        pass

    def _detach_object_from_gripper(self):
        """Detach object from gripper."""
        pass


def main(args=None):
    rclpy.init(args=args)
    server = PickPlaceServer()

    # Example usage
    object_pose = Pose()
    object_pose.position.x = 0.5
    object_pose.position.y = 0.0
    object_pose.position.z = 0.75  # On table
    object_pose.orientation.w = 1.0

    place_pose = Pose()
    place_pose.position.x = 0.5
    place_pose.position.y = 0.3
    place_pose.position.z = 0.75
    place_pose.orientation.w = 1.0

    # Pick
    if server.pick('block_1', object_pose, (0.05, 0.05, 0.05)):
        # Place
        server.place('block_1', place_pose)

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Grasp Quality Metrics

```python
class GraspQualityMetrics:
    """
    Evaluate grasp quality.

    Computes various metrics for grasp ranking.
    """

    @staticmethod
    def force_closure_metric(contact_points: np.ndarray,
                            contact_normals: np.ndarray) -> float:
        """
        Compute force closure quality.

        Force closure means the grasp can resist any external wrench.

        Args:
            contact_points: Nx3 array of contact positions
            contact_normals: Nx3 array of contact normals (inward)

        Returns:
            Quality score (higher is better)
        """
        # Build grasp matrix
        n = len(contact_points)
        G = np.zeros((6, n))

        for i in range(n):
            # Force contribution
            G[:3, i] = contact_normals[i]
            # Torque contribution
            G[3:, i] = np.cross(contact_points[i], contact_normals[i])

        # Compute minimum singular value
        # Higher value = better force closure
        _, s, _ = np.linalg.svd(G)
        return s[-1] if len(s) > 0 else 0.0

    @staticmethod
    def grasp_width_score(grasp_width: float, object_width: float,
                          gripper_max: float) -> float:
        """
        Score based on how well gripper fits object.

        Args:
            grasp_width: Required grasp aperture
            object_width: Object dimension being grasped
            gripper_max: Maximum gripper opening

        Returns:
            Score 0-1 (1 is perfect fit)
        """
        if grasp_width > gripper_max:
            return 0.0  # Can't grasp

        # Prefer grasps using middle of gripper range
        utilization = grasp_width / gripper_max
        return 1.0 - abs(0.5 - utilization)

    @staticmethod
    def approach_clearance_score(approach_direction: np.ndarray,
                                 obstacles: list) -> float:
        """
        Score based on approach clearance.

        Args:
            approach_direction: Unit vector of approach
            obstacles: List of obstacle positions

        Returns:
            Score 0-1 (1 is clear approach)
        """
        # Simplified - would do actual collision checking
        return 1.0

    @staticmethod
    def stability_score(grasp_pose: Pose, object_com: np.ndarray) -> float:
        """
        Score based on grasp stability.

        Prefers grasps below center of mass for stability.

        Args:
            grasp_pose: Grasp contact position
            object_com: Object center of mass

        Returns:
            Score 0-1
        """
        grasp_z = grasp_pose.position.z
        com_z = object_com[2]

        if grasp_z <= com_z:
            return 1.0  # Grasp at or below COM
        else:
            return max(0, 1.0 - (grasp_z - com_z))
```

## Error Handling and Recovery

```python
class ManipulationRecovery:
    """
    Recovery behaviors for manipulation failures.
    """

    def __init__(self, pick_place: PickPlaceServer):
        self.pick_place = pick_place
        self.max_retries = 3

    def handle_grasp_failure(self, object_name: str,
                             object_pose: Pose) -> bool:
        """
        Handle failed grasp attempt.

        Strategies:
        1. Retry with different grasp
        2. Redetect object pose
        3. Adjust approach angle
        """
        for retry in range(self.max_retries):
            # Strategy 1: Try alternative grasp
            if retry == 0:
                # Re-plan with different grasp index
                pass

            # Strategy 2: Redetect object
            elif retry == 1:
                # object_pose = self.detect_object(object_name)
                pass

            # Strategy 3: Gentle touch to verify contact
            elif retry == 2:
                # Slow approach with force feedback
                pass

        return False

    def handle_drop(self, object_name: str,
                    last_known_pose: Pose) -> bool:
        """
        Handle dropped object.

        1. Open gripper fully
        2. Move to safe position
        3. Redetect object
        4. Retry pick
        """
        # Open gripper
        self.pick_place._gripper.open()

        # Move to safe position
        # safe_pose = ...
        # self.pick_place._plan_and_execute(safe_pose)

        # Would integrate with perception to find dropped object
        return False

    def handle_collision(self) -> bool:
        """
        Handle unexpected collision.

        1. Stop motion
        2. Open gripper if holding object
        3. Back away carefully
        """
        # Emergency stop would be hardware-level
        # This handles software recovery

        # Open gripper to release any object
        self.pick_place._gripper.open()

        # Plan retreat motion
        return False
```

## Summary

Key takeaways from this lesson:

1. **Pick and place** is a multi-step coordinated process
2. **Grasp planning** generates and ranks grasp candidates
3. **Gripper control** requires force and position management
4. **Quality metrics** help select the best grasp
5. **Recovery behaviors** handle inevitable failures

## Next Steps

Continue to [Chapter 6: LLM Integration](../chapter-06-llm-integration/lesson-01-llm-basics.md) to learn:
- Connecting language models to robots
- Natural language command interpretation
- Task planning with LLMs

## Additional Resources

- [Grasp Planning Tutorial](https://manipulation.csail.mit.edu/)
- [MoveIt Pick and Place](https://moveit.picknik.ai/main/doc/examples/pick_place/pick_place_tutorial.html)
- [Dexterous Manipulation](https://arxiv.org/abs/2203.13251)
