---
sidebar_position: 2
title: "Lesson 2: MoveIt2 Motion Planning"
description: "Planning and executing robot arm trajectories with MoveIt2"
---

# MoveIt2 Motion Planning

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand MoveIt2 architecture and components
2. Configure MoveIt2 for a humanoid arm
3. Plan and execute collision-free trajectories
4. Use different planning algorithms

## Prerequisites

- Completed Lesson 1 (Kinematics)
- ROS 2 and MoveIt2 installed
- URDF model of your robot arm

## What is MoveIt2?

MoveIt2 is the motion planning framework for ROS 2.

```
┌─────────────────────────────────────────────────────────────┐
│                   MoveIt2 Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Application                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Move Group Interface                    │   │
│  │         (High-level motion planning API)            │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Planning Pipeline                       │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │   │   OMPL     │  │   PILZ      │  │   STOMP   │  │   │
│  │   │  Planners  │  │  Industrial │  │   (Opt.)  │  │   │
│  │   └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Planning Scene                          │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │   │   Robot    │  │   World     │  │  Allowed  │  │   │
│  │   │   State    │  │  Geometry   │  │ Collision │  │   │
│  │   └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Trajectory Execution                       │   │
│  │              (Controller Interface)                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installing MoveIt2

```bash
# Install MoveIt2 for ROS 2 Humble
sudo apt install ros-humble-moveit

# Install development tools
sudo apt install ros-humble-moveit-setup-assistant
```

## MoveIt2 Configuration

### Using Setup Assistant

The MoveIt Setup Assistant generates configuration:

```bash
# Launch setup assistant
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```

### Manual Configuration

#### SRDF (Semantic Robot Description)

```xml
<?xml version="1.0" ?>
<!--
  Humanoid Arm SRDF
  Physical AI Book - Chapter 5

  Defines planning groups, end effectors, and virtual joints.
-->
<robot name="humanoid">

  <!-- Arm Planning Group -->
  <group name="arm">
    <chain base_link="shoulder_link" tip_link="gripper_link" />
  </group>

  <!-- Gripper Planning Group -->
  <group name="gripper">
    <joint name="gripper_left_finger_joint" />
    <joint name="gripper_right_finger_joint" />
  </group>

  <!-- End Effector Definition -->
  <end_effector name="gripper"
                parent_link="wrist_roll_link"
                group="gripper" />

  <!-- Named Poses -->
  <group_state name="home" group="arm">
    <joint name="shoulder_pitch_joint" value="0" />
    <joint name="shoulder_roll_joint" value="0" />
    <joint name="shoulder_yaw_joint" value="0" />
    <joint name="elbow_pitch_joint" value="0" />
    <joint name="wrist_yaw_joint" value="0" />
    <joint name="wrist_pitch_joint" value="0" />
    <joint name="wrist_roll_joint" value="0" />
  </group_state>

  <group_state name="ready" group="arm">
    <joint name="shoulder_pitch_joint" value="0.5" />
    <joint name="shoulder_roll_joint" value="0.2" />
    <joint name="shoulder_yaw_joint" value="0" />
    <joint name="elbow_pitch_joint" value="-1.57" />
    <joint name="wrist_yaw_joint" value="0" />
    <joint name="wrist_pitch_joint" value="0" />
    <joint name="wrist_roll_joint" value="0" />
  </group_state>

  <!-- Disable Collisions -->
  <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent" />
  <disable_collisions link1="upper_arm_link" link2="forearm_link" reason="Adjacent" />
  <disable_collisions link1="forearm_link" link2="wrist_link" reason="Adjacent" />

</robot>
```

#### Planning Configuration (moveit.yaml)

```yaml
# MoveIt2 Configuration
# Physical AI Book - Chapter 5

move_group:
  ros__parameters:
    # Planning parameters
    planning_scene_monitor:
      publish_planning_scene: true
      publish_geometry_updates: true
      publish_state_updates: true
      publish_transforms_updates: true

    # Move Group capabilities
    capabilities:
      - move_group/MoveGroupCartesianPathService
      - move_group/MoveGroupKinematicsService
      - move_group/MoveGroupMoveAction
      - move_group/MoveGroupPlanService
      - move_group/MoveGroupQueryPlannersService

    # Planning pipeline
    planning_pipelines:
      pipeline_names: ["ompl", "pilz_industrial_motion_planner"]

    default_planning_pipeline: ompl

    # OMPL Planning Configuration
    ompl:
      planning_plugin: ompl_interface/OMPLPlanner
      request_adapters:
        - default_planner_request_adapters/AddTimeOptimalParameterization
        - default_planner_request_adapters/FixWorkspaceBounds
        - default_planner_request_adapters/FixStartStateBounds
        - default_planner_request_adapters/FixStartStateCollision
        - default_planner_request_adapters/FixStartStatePathConstraints

      # Start state fixing
      start_state_max_bounds_error: 0.1

    # Pilz Industrial Planner
    pilz_industrial_motion_planner:
      planning_plugin: pilz_industrial_motion_planner/CommandPlanner
      request_adapters: ""
      default_planner_config: PTP

# OMPL Planner Configuration
ompl_planning:
  ros__parameters:
    arm:
      default_planner_config: RRTConnect
      planner_configs:
        RRTConnect:
          type: geometric::RRTConnect
          range: 0.0

        RRTstar:
          type: geometric::RRTstar
          range: 0.0
          goal_bias: 0.05
          delay_collision_checking: 1

        PRM:
          type: geometric::PRM
          max_nearest_neighbors: 10

        BiTRRT:
          type: geometric::BiTRRT
          range: 0.0
          temp_change_factor: 0.1

      projection_evaluator: joints(shoulder_pitch_joint,elbow_pitch_joint)
      longest_valid_segment_fraction: 0.01

# Trajectory Execution
trajectory_execution:
  ros__parameters:
    moveit_manage_controllers: true
    trajectory_execution:
      allowed_execution_duration_scaling: 1.2
      allowed_goal_duration_margin: 0.5
      allowed_start_tolerance: 0.01

    moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager

    controller_names:
      - arm_controller

    arm_controller:
      type: FollowJointTrajectory
      action_ns: follow_joint_trajectory
      joints:
        - shoulder_pitch_joint
        - shoulder_roll_joint
        - shoulder_yaw_joint
        - elbow_pitch_joint
        - wrist_yaw_joint
        - wrist_pitch_joint
        - wrist_roll_joint
```

## Using MoveIt2 in Python

### Basic Motion Planning

```python
#!/usr/bin/env python3
"""
MoveIt2 Motion Planning Example
Physical AI Book - Chapter 5

Demonstrates basic motion planning with MoveIt2.

Usage:
    ros2 run physical_ai_examples moveit_planner

Expected Output:
    [INFO] [moveit_planner]: Planning to target pose...
    [INFO] [moveit_planner]: Plan found! Executing...
    [INFO] [moveit_planner]: Motion complete

Dependencies:
    - rclpy
    - moveit_py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
import numpy as np


class MoveItPlanner(Node):
    """MoveIt2 motion planning node."""

    def __init__(self):
        super().__init__('moveit_planner')

        # Initialize MoveIt
        self.moveit = MoveItPy(node_name="moveit_py")
        self.arm = self.moveit.get_planning_component("arm")
        self.planning_scene = self.moveit.get_planning_scene_monitor()

        self.get_logger().info('MoveIt2 planner ready')

    def plan_to_pose(self, pose: Pose) -> bool:
        """
        Plan motion to target pose.

        Args:
            pose: Target end effector pose

        Returns:
            True if planning succeeded
        """
        # Create pose stamped
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = pose

        # Set target
        self.arm.set_goal_state(pose_stamped=pose_stamped)

        # Plan
        self.get_logger().info('Planning to target pose...')
        plan_result = self.arm.plan()

        if plan_result:
            self.get_logger().info('Plan found!')
            return True
        else:
            self.get_logger().error('Planning failed')
            return False

    def plan_to_named_target(self, name: str) -> bool:
        """
        Plan to a named configuration.

        Args:
            name: Named target (e.g., 'home', 'ready')

        Returns:
            True if planning succeeded
        """
        self.arm.set_goal_state(configuration_name=name)
        return self.arm.plan() is not None

    def execute(self) -> bool:
        """Execute the planned trajectory."""
        self.get_logger().info('Executing plan...')
        result = self.moveit.execute(
            self.arm.get_robot_trajectory(),
            controllers=[]  # Use default controllers
        )
        self.get_logger().info('Motion complete')
        return result

    def plan_cartesian_path(self, waypoints: list,
                            step: float = 0.01,
                            jump_threshold: float = 0.0) -> tuple:
        """
        Plan a Cartesian path through waypoints.

        Args:
            waypoints: List of Pose objects
            step: Maximum step in meters
            jump_threshold: Max joint jump (0 disables)

        Returns:
            (trajectory, fraction) tuple
        """
        trajectory, fraction = self.arm.compute_cartesian_path(
            waypoints=waypoints,
            max_step=step,
            jump_threshold=jump_threshold
        )

        self.get_logger().info(f'Cartesian path: {fraction*100:.1f}% achieved')
        return trajectory, fraction


def main(args=None):
    rclpy.init(args=args)

    planner = MoveItPlanner()

    # Example: Plan to pose
    target = Pose()
    target.position.x = 0.4
    target.position.y = 0.0
    target.position.z = 0.3
    target.orientation.w = 1.0

    if planner.plan_to_pose(target):
        planner.execute()

    # Plan to named target
    if planner.plan_to_named_target('home'):
        planner.execute()

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Planning Scene Management

```python
class PlanningSceneManager:
    """
    Manage the MoveIt planning scene.

    Add obstacles, attach/detach objects from the robot.
    """

    def __init__(self, moveit: MoveItPy):
        self.planning_scene = moveit.get_planning_scene_monitor()

    def add_box(self, name: str, pose: Pose,
                dimensions: tuple, frame_id: str = "base_link"):
        """
        Add a box obstacle to the scene.

        Args:
            name: Unique object name
            pose: Box pose
            dimensions: (x, y, z) dimensions in meters
            frame_id: Reference frame
        """
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = name

        # Define box primitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(dimensions)

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        self.planning_scene.apply_collision_object(collision_object)

    def add_table(self, height: float = 0.75,
                  size: tuple = (1.0, 2.0, 0.02)):
        """Add a table to the scene."""
        pose = Pose()
        pose.position.x = 0.6
        pose.position.y = 0.0
        pose.position.z = height - size[2] / 2
        pose.orientation.w = 1.0

        self.add_box("table", pose, size)

    def attach_object(self, object_name: str, link_name: str,
                      touch_links: list = None):
        """
        Attach an object to the robot.

        Args:
            object_name: Name of object in scene
            link_name: Robot link to attach to
            touch_links: Links allowed to touch object
        """
        from moveit_msgs.msg import AttachedCollisionObject

        attached = AttachedCollisionObject()
        attached.link_name = link_name
        attached.object.id = object_name
        attached.object.operation = attached.object.ADD

        if touch_links:
            attached.touch_links = touch_links

        self.planning_scene.apply_attached_collision_object(attached)

    def detach_object(self, object_name: str, link_name: str):
        """Detach an object from the robot."""
        from moveit_msgs.msg import AttachedCollisionObject

        attached = AttachedCollisionObject()
        attached.link_name = link_name
        attached.object.id = object_name
        attached.object.operation = attached.object.REMOVE

        self.planning_scene.apply_attached_collision_object(attached)

    def remove_object(self, name: str):
        """Remove an object from the scene."""
        from moveit_msgs.msg import CollisionObject

        collision_object = CollisionObject()
        collision_object.id = name
        collision_object.operation = CollisionObject.REMOVE

        self.planning_scene.apply_collision_object(collision_object)

    def clear_scene(self):
        """Remove all objects from the scene."""
        self.planning_scene.clear()
```

## Planning Algorithms

### OMPL Planners

| Planner | Description | Best For |
|---------|-------------|----------|
| **RRTConnect** | Bidirectional RRT | General motion planning |
| **RRT*** | Optimal RRT | Path quality matters |
| **PRM** | Probabilistic Roadmap | Repeated queries |
| **BiTRRT** | Bidirectional T-RRT | Cost-aware planning |
| **LazyPRM** | Lazy collision checking | Complex environments |

### Pilz Industrial Planner

For simple, predictable motions:

```python
def plan_linear_motion(self, target: Pose) -> bool:
    """
    Plan a linear (Cartesian) motion.

    Uses Pilz industrial planner for straight-line paths.
    """
    # Set planning pipeline
    self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
    self.arm.set_planner_id("LIN")  # Linear motion

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "base_link"
    pose_stamped.pose = target

    self.arm.set_goal_state(pose_stamped=pose_stamped)

    return self.arm.plan() is not None


def plan_circular_motion(self, center: Pose, target: Pose) -> bool:
    """
    Plan a circular arc motion.

    Uses Pilz CIRC planner.
    """
    self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
    self.arm.set_planner_id("CIRC")

    # CIRC requires auxiliary point (center)
    # Implementation depends on MoveIt2 version
    pass
```

## Velocity and Acceleration Scaling

```python
def set_velocity_scaling(self, scale: float):
    """
    Set velocity scaling factor.

    Args:
        scale: 0.0 to 1.0 (fraction of max velocity)
    """
    self.arm.set_max_velocity_scaling_factor(scale)


def set_acceleration_scaling(self, scale: float):
    """
    Set acceleration scaling factor.

    Args:
        scale: 0.0 to 1.0 (fraction of max acceleration)
    """
    self.arm.set_max_acceleration_scaling_factor(scale)


# Example: Slow, precise motion
planner.set_velocity_scaling(0.1)
planner.set_acceleration_scaling(0.1)

# Example: Fast motion
planner.set_velocity_scaling(0.8)
planner.set_acceleration_scaling(0.5)
```

## Constraints

### Position Constraints

```python
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive


def add_position_constraint(self, link_name: str,
                            region_pose: Pose,
                            region_dimensions: tuple):
    """
    Constrain a link to stay within a region.

    Args:
        link_name: Link to constrain
        region_pose: Center of constraint region
        region_dimensions: (x, y, z) region size
    """
    constraint = PositionConstraint()
    constraint.header.frame_id = "base_link"
    constraint.link_name = link_name

    # Define constraint region as a box
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = list(region_dimensions)

    constraint.constraint_region.primitives.append(box)
    constraint.constraint_region.primitive_poses.append(region_pose)
    constraint.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(constraint)

    self.arm.set_path_constraints(constraints)
```

### Orientation Constraints

```python
from moveit_msgs.msg import OrientationConstraint


def add_orientation_constraint(self, link_name: str,
                               orientation: tuple,
                               tolerance: tuple = (0.1, 0.1, 0.1)):
    """
    Constrain end effector orientation.

    Args:
        link_name: Link to constrain
        orientation: (x, y, z, w) quaternion
        tolerance: (roll, pitch, yaw) tolerance in radians
    """
    constraint = OrientationConstraint()
    constraint.header.frame_id = "base_link"
    constraint.link_name = link_name

    constraint.orientation.x = orientation[0]
    constraint.orientation.y = orientation[1]
    constraint.orientation.z = orientation[2]
    constraint.orientation.w = orientation[3]

    constraint.absolute_x_axis_tolerance = tolerance[0]
    constraint.absolute_y_axis_tolerance = tolerance[1]
    constraint.absolute_z_axis_tolerance = tolerance[2]
    constraint.weight = 1.0

    constraints = Constraints()
    constraints.orientation_constraints.append(constraint)

    self.arm.set_path_constraints(constraints)
```

## Summary

Key takeaways from this lesson:

1. **MoveIt2** provides complete motion planning
2. **SRDF** defines semantic robot information
3. **OMPL planners** handle complex environments
4. **Pilz planners** provide industrial-style motion
5. **Constraints** enable task-specific planning

## Next Steps

In the [next lesson](./lesson-03-grasping.md), we will:
- Implement grasp planning
- Use parallel gripper control
- Execute pick and place operations

## Additional Resources

- [MoveIt2 Documentation](https://moveit.picknik.ai/main/index.html)
- [OMPL Library](https://ompl.kavrakilab.org/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html)
