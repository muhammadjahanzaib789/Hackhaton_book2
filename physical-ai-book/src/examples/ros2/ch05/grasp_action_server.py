#!/usr/bin/env python3
"""
Grasp Action Server
Physical AI Book - Chapter 5: Manipulation

ROS 2 action server for pick and place operations with
grasp planning and execution.

Usage:
    ros2 run physical_ai_examples grasp_action_server

    # Send a pick goal
    ros2 action send_goal /pick_place physical_ai_msgs/action/PickPlace \
        "{object_id: 'cube_1', action: 'pick'}"

Expected Output:
    [INFO] [grasp_server]: Grasp action server ready
    [INFO] [grasp_server]: Received pick request for cube_1
    [INFO] [grasp_server]: Generated 4 grasp candidates
    [INFO] [grasp_server]: Executing grasp 1/4
    [INFO] [grasp_server]: Opening gripper...
    [INFO] [grasp_server]: Approaching object...
    [INFO] [grasp_server]: Closing gripper...
    [INFO] [grasp_server]: Lifting object...
    [INFO] [grasp_server]: Pick succeeded!

Dependencies:
    - rclpy
    - geometry_msgs
    - sensor_msgs
    - control_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, PoseStamped, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float64MultiArray
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time


class GraspStatus(Enum):
    """Grasp operation status."""
    SUCCESS = 0
    PLANNING_FAILED = 1
    APPROACH_FAILED = 2
    GRASP_FAILED = 3
    LIFT_FAILED = 4
    PLACE_FAILED = 5
    OBJECT_DROPPED = 6


@dataclass
class Grasp:
    """Grasp configuration."""
    pre_grasp_pose: Pose
    grasp_pose: Pose
    post_grasp_pose: Pose
    grasp_width: float
    approach_direction: np.ndarray
    score: float


class GraspGenerator:
    """
    Generate grasp candidates for objects.

    Supports rectangular objects with top-down grasps.
    """

    def __init__(self, approach_distance: float = 0.10,
                 lift_height: float = 0.05):
        """
        Args:
            approach_distance: Distance to pre-grasp position
            lift_height: Height to lift after grasping
        """
        self.approach_distance = approach_distance
        self.lift_height = lift_height

    def generate(self, object_pose: Pose,
                 object_dimensions: Tuple[float, float, float],
                 gripper_max_width: float = 0.08) -> List[Grasp]:
        """
        Generate grasp candidates for a box-shaped object.

        Args:
            object_pose: Object pose (center of mass)
            object_dimensions: (length, width, height) in meters
            gripper_max_width: Maximum gripper opening

        Returns:
            List of Grasp objects sorted by score
        """
        grasps = []
        length, width, height = object_dimensions

        # Top-down grasps from different angles
        angles = [0, np.pi/2, np.pi, -np.pi/2]
        widths = [width, length, width, length]

        for yaw, grasp_width in zip(angles, widths):
            # Skip if gripper can't open wide enough
            if grasp_width > gripper_max_width:
                continue

            # Create grasp poses
            grasp_pose = self._create_grasp_pose(object_pose, height, yaw)
            pre_grasp = self._offset_pose(grasp_pose, 0, 0, self.approach_distance)
            post_grasp = self._offset_pose(grasp_pose, 0, 0, self.lift_height)

            # Score based on grasp width utilization
            score = 1.0 - abs(0.5 - grasp_width / gripper_max_width)

            grasps.append(Grasp(
                pre_grasp_pose=pre_grasp,
                grasp_pose=grasp_pose,
                post_grasp_pose=post_grasp,
                grasp_width=grasp_width,
                approach_direction=np.array([0, 0, -1]),
                score=score
            ))

        # Sort by score (highest first)
        grasps.sort(key=lambda g: g.score, reverse=True)

        return grasps

    def _create_grasp_pose(self, object_pose: Pose,
                           height: float, yaw: float) -> Pose:
        """Create grasp pose at object top."""
        pose = Pose()

        pose.position.x = object_pose.position.x
        pose.position.y = object_pose.position.y
        pose.position.z = object_pose.position.z + height / 2

        # Orientation: gripper pointing down, rotated by yaw
        # Quaternion for rotation around Z then X (pointing down)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        cx = np.cos(np.pi / 2)  # 90 degrees pitch
        sx = np.sin(np.pi / 2)

        pose.orientation.x = cx * cy
        pose.orientation.y = cx * sy
        pose.orientation.z = sx * sy
        pose.orientation.w = sx * cy

        return pose

    def _offset_pose(self, pose: Pose, dx: float, dy: float, dz: float) -> Pose:
        """Offset a pose in the gripper frame."""
        new_pose = Pose()
        new_pose.position.x = pose.position.x + dx
        new_pose.position.y = pose.position.y + dy
        new_pose.position.z = pose.position.z + dz
        new_pose.orientation = pose.orientation
        return new_pose


class GripperController:
    """
    Simulated gripper controller.

    In production, would interface with real hardware via ROS 2.
    """

    def __init__(self, node: Node):
        self.node = node
        self.max_width = 0.08
        self.current_width = self.max_width
        self.is_gripping = False

        # Publisher for gripper commands
        self.cmd_pub = node.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10
        )

    def open(self, width: Optional[float] = None) -> bool:
        """Open gripper to specified width."""
        target_width = width if width else self.max_width
        self.node.get_logger().info(f'Opening gripper to {target_width*100:.1f}cm')

        msg = Float64MultiArray()
        msg.data = [target_width / 2, target_width / 2]
        self.cmd_pub.publish(msg)

        time.sleep(0.5)  # Simulate motion time
        self.current_width = target_width
        self.is_gripping = False

        return True

    def close(self, target_width: float = 0.0, max_effort: float = 50.0) -> bool:
        """Close gripper to grasp object."""
        self.node.get_logger().info(f'Closing gripper (effort: {max_effort}N)')

        msg = Float64MultiArray()
        msg.data = [target_width / 2, target_width / 2]
        self.cmd_pub.publish(msg)

        time.sleep(0.5)

        # Simulate grasp detection (would use force feedback)
        if target_width < self.current_width:
            self.is_gripping = True
            self.current_width = target_width + 0.01  # Object width

        return self.is_gripping


class MotionPlanner:
    """
    Simulated motion planner.

    In production, would interface with MoveIt2.
    """

    def __init__(self, node: Node):
        self.node = node

        # Joint command publisher
        self.joint_pub = node.create_publisher(
            JointState, '/arm_controller/command', 10
        )

    def plan_to_pose(self, target: Pose) -> bool:
        """Plan motion to target pose."""
        self.node.get_logger().info(
            f'Planning to ({target.position.x:.2f}, '
            f'{target.position.y:.2f}, {target.position.z:.2f})'
        )
        time.sleep(0.3)  # Simulate planning time
        return True

    def execute(self) -> bool:
        """Execute planned trajectory."""
        self.node.get_logger().info('Executing trajectory')
        time.sleep(0.5)  # Simulate motion time
        return True

    def plan_cartesian(self, waypoints: List[Pose]) -> bool:
        """Plan Cartesian path through waypoints."""
        self.node.get_logger().info(
            f'Planning Cartesian path ({len(waypoints)} waypoints)'
        )
        time.sleep(0.2)
        return True


class GraspActionServer(Node):
    """
    ROS 2 action server for pick and place operations.

    Coordinates grasp planning, motion planning, and gripper control.
    """

    def __init__(self):
        super().__init__('grasp_server')

        self._callback_group = ReentrantCallbackGroup()

        # Initialize components
        self.grasp_generator = GraspGenerator()
        self.gripper = GripperController(self)
        self.planner = MotionPlanner(self)

        # Object database (in production, would come from perception)
        self.objects: Dict[str, dict] = {
            'cube_1': {
                'pose': self._make_pose(0.5, 0.0, 0.75),
                'dimensions': (0.05, 0.05, 0.05)
            },
            'box_1': {
                'pose': self._make_pose(0.5, 0.2, 0.75),
                'dimensions': (0.08, 0.04, 0.03)
            }
        }

        # Currently held object
        self.held_object: Optional[str] = None

        # Subscribers for object updates
        self.object_sub = self.create_subscription(
            PoseStamped, '/detected_object',
            self._object_callback, 10
        )

        self.get_logger().info('Grasp action server ready')

    def pick(self, object_id: str) -> GraspStatus:
        """
        Execute pick operation.

        Args:
            object_id: ID of object to pick

        Returns:
            GraspStatus indicating success or failure mode
        """
        self.get_logger().info(f'Received pick request for {object_id}')

        # Get object info
        if object_id not in self.objects:
            self.get_logger().error(f'Unknown object: {object_id}')
            return GraspStatus.PLANNING_FAILED

        obj = self.objects[object_id]

        # Generate grasp candidates
        grasps = self.grasp_generator.generate(
            obj['pose'],
            obj['dimensions'],
            self.gripper.max_width
        )

        if not grasps:
            self.get_logger().error('No valid grasps found')
            return GraspStatus.PLANNING_FAILED

        self.get_logger().info(f'Generated {len(grasps)} grasp candidates')

        # Try each grasp
        for i, grasp in enumerate(grasps):
            self.get_logger().info(f'Executing grasp {i+1}/{len(grasps)}')

            status = self._execute_pick(grasp, grasp.grasp_width)

            if status == GraspStatus.SUCCESS:
                self.held_object = object_id
                return GraspStatus.SUCCESS

        return GraspStatus.GRASP_FAILED

    def place(self, target_pose: Pose) -> GraspStatus:
        """
        Execute place operation.

        Args:
            target_pose: Target placement pose

        Returns:
            GraspStatus indicating success or failure
        """
        if self.held_object is None:
            self.get_logger().error('No object in gripper')
            return GraspStatus.PLANNING_FAILED

        self.get_logger().info(f'Placing {self.held_object}')

        status = self._execute_place(target_pose)

        if status == GraspStatus.SUCCESS:
            self.held_object = None

        return status

    def _execute_pick(self, grasp: Grasp, object_width: float) -> GraspStatus:
        """Execute the pick sequence."""
        try:
            # 1. Open gripper
            self.get_logger().info('Opening gripper...')
            if not self.gripper.open():
                return GraspStatus.GRASP_FAILED

            # 2. Move to pre-grasp
            self.get_logger().info('Approaching object...')
            if not self.planner.plan_to_pose(grasp.pre_grasp_pose):
                return GraspStatus.APPROACH_FAILED
            if not self.planner.execute():
                return GraspStatus.APPROACH_FAILED

            # 3. Move to grasp pose (Cartesian)
            if not self.planner.plan_cartesian([grasp.grasp_pose]):
                return GraspStatus.APPROACH_FAILED
            if not self.planner.execute():
                return GraspStatus.APPROACH_FAILED

            # 4. Close gripper
            self.get_logger().info('Closing gripper...')
            if not self.gripper.close(target_width=object_width * 0.9):
                return GraspStatus.GRASP_FAILED

            # 5. Lift
            self.get_logger().info('Lifting object...')
            if not self.planner.plan_cartesian([grasp.post_grasp_pose]):
                return GraspStatus.LIFT_FAILED
            if not self.planner.execute():
                return GraspStatus.LIFT_FAILED

            self.get_logger().info('Pick succeeded!')
            return GraspStatus.SUCCESS

        except Exception as e:
            self.get_logger().error(f'Pick failed: {e}')
            return GraspStatus.GRASP_FAILED

    def _execute_place(self, target_pose: Pose) -> GraspStatus:
        """Execute the place sequence."""
        try:
            # Create place poses
            pre_place = self._offset_pose_z(target_pose, 0.1)
            post_place = self._offset_pose_z(target_pose, 0.1)

            # 1. Move to pre-place
            self.get_logger().info('Moving to place position...')
            if not self.planner.plan_to_pose(pre_place):
                return GraspStatus.PLACE_FAILED
            if not self.planner.execute():
                return GraspStatus.PLACE_FAILED

            # 2. Lower to place
            self.get_logger().info('Lowering object...')
            if not self.planner.plan_cartesian([target_pose]):
                return GraspStatus.PLACE_FAILED
            if not self.planner.execute():
                return GraspStatus.PLACE_FAILED

            # 3. Open gripper
            self.get_logger().info('Releasing object...')
            if not self.gripper.open():
                return GraspStatus.PLACE_FAILED

            # 4. Retreat
            self.get_logger().info('Retreating...')
            if not self.planner.plan_cartesian([post_place]):
                return GraspStatus.PLACE_FAILED
            if not self.planner.execute():
                return GraspStatus.PLACE_FAILED

            self.get_logger().info('Place succeeded!')
            return GraspStatus.SUCCESS

        except Exception as e:
            self.get_logger().error(f'Place failed: {e}')
            return GraspStatus.PLACE_FAILED

    def _object_callback(self, msg: PoseStamped):
        """Update object pose from detection."""
        object_id = msg.header.frame_id
        if object_id in self.objects:
            self.objects[object_id]['pose'] = msg.pose

    def _make_pose(self, x: float, y: float, z: float) -> Pose:
        """Create a Pose with default orientation."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        return pose

    def _offset_pose_z(self, pose: Pose, dz: float) -> Pose:
        """Offset pose in Z direction."""
        new_pose = Pose()
        new_pose.position.x = pose.position.x
        new_pose.position.y = pose.position.y
        new_pose.position.z = pose.position.z + dz
        new_pose.orientation = pose.orientation
        return new_pose


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    server = GraspActionServer()

    # Demo: Pick and place sequence
    try:
        # Pick cube_1
        status = server.pick('cube_1')

        if status == GraspStatus.SUCCESS:
            # Place at new location
            place_pose = Pose()
            place_pose.position.x = 0.5
            place_pose.position.y = 0.3
            place_pose.position.z = 0.78
            place_pose.orientation.w = 1.0

            server.place(place_pose)

        rclpy.spin(server)

    except KeyboardInterrupt:
        pass

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
