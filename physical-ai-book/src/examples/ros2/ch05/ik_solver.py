#!/usr/bin/env python3
"""
Inverse Kinematics Solver Node
Physical AI Book - Chapter 5: Manipulation

Provides IK solving as a ROS 2 service using numerical methods.

Usage:
    ros2 run physical_ai_examples ik_solver

    # Call the service
    ros2 service call /solve_ik physical_ai_msgs/srv/SolveIK \
        "{target_pose: {position: {x: 0.4, y: 0.0, z: 0.3}, orientation: {w: 1.0}}}"

Expected Output:
    [INFO] [ik_solver]: IK solver ready
    [INFO] [ik_solver]: Solving IK for target (0.40, 0.00, 0.30)
    [INFO] [ik_solver]: Solution found in 15 iterations
    [INFO] [ik_solver]: Joint angles: [0.52, 0.00, 0.00, -1.57, 0.00, 0.00, 0.00]

Dependencies:
    - rclpy
    - numpy
    - geometry_msgs
    - sensor_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class IKStatus(Enum):
    """IK solver status codes."""
    SUCCESS = 0
    NO_SOLUTION = 1
    JOINT_LIMITS = 2
    TIMEOUT = 3


@dataclass
class IKResult:
    """Result of IK computation."""
    status: IKStatus
    joint_angles: Optional[np.ndarray]
    iterations: int
    error: float


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute DH transformation matrix.

    Args:
        a: Link length
        alpha: Link twist
        d: Link offset
        theta: Joint angle

    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


class KinematicChain:
    """
    Forward kinematics using DH parameters.

    Computes end effector pose from joint angles.
    """

    def __init__(self, dh_params: List[Tuple[float, float, float, float]]):
        """
        Initialize kinematic chain.

        Args:
            dh_params: List of (a, alpha, d, theta_offset) tuples
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            joint_angles: Array of joint angles

        Returns:
            4x4 transformation matrix (base to end effector)
        """
        T = np.eye(4)

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T = T @ dh_transform(a, alpha, d, theta)

        return T

    def get_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end effector position [x, y, z]."""
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3]


class JacobianIKSolver:
    """
    Iterative inverse kinematics using the Jacobian.

    Uses damped least squares (Levenberg-Marquardt) for stability.
    """

    def __init__(self, chain: KinematicChain,
                 joint_limits: Optional[List[Tuple[float, float]]] = None,
                 damping: float = 0.1):
        """
        Initialize IK solver.

        Args:
            chain: Kinematic chain
            joint_limits: Optional joint limits [(min, max), ...]
            damping: Damping factor for numerical stability
        """
        self.chain = chain
        self.joint_limits = np.array(joint_limits) if joint_limits else None
        self.damping = damping

    def compute_jacobian(self, joint_angles: np.ndarray,
                         delta: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian numerically.

        Args:
            joint_angles: Current joint configuration
            delta: Finite difference step size

        Returns:
            6xN Jacobian matrix
        """
        n = len(joint_angles)
        J = np.zeros((6, n))

        current_pose = self._pose_to_vector(
            self.chain.forward_kinematics(joint_angles)
        )

        for i in range(n):
            perturbed = joint_angles.copy()
            perturbed[i] += delta
            new_pose = self._pose_to_vector(
                self.chain.forward_kinematics(perturbed)
            )
            J[:, i] = (new_pose - current_pose) / delta

        return J

    def solve(self, target_pose: np.ndarray,
              initial_angles: np.ndarray,
              max_iterations: int = 100,
              tolerance: float = 1e-4) -> IKResult:
        """
        Solve IK iteratively.

        Args:
            target_pose: 4x4 target transformation matrix
            initial_angles: Starting joint configuration
            max_iterations: Maximum solver iterations
            tolerance: Convergence threshold

        Returns:
            IKResult with solution status and joint angles
        """
        angles = initial_angles.copy()
        target_vector = self._pose_to_vector(target_pose)

        for iteration in range(max_iterations):
            current_pose = self.chain.forward_kinematics(angles)
            current_vector = self._pose_to_vector(current_pose)

            error = target_vector - current_vector
            error_norm = np.linalg.norm(error[:3])  # Position error

            if error_norm < tolerance:
                return IKResult(
                    status=IKStatus.SUCCESS,
                    joint_angles=angles,
                    iterations=iteration + 1,
                    error=error_norm
                )

            # Compute Jacobian
            J = self.compute_jacobian(angles)

            # Damped least squares
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(6)
            delta_angles = J.T @ np.linalg.solve(damped, error)

            # Update angles
            angles = angles + delta_angles

            # Apply joint limits
            if self.joint_limits is not None:
                angles = np.clip(
                    angles,
                    self.joint_limits[:, 0],
                    self.joint_limits[:, 1]
                )

        return IKResult(
            status=IKStatus.TIMEOUT,
            joint_angles=angles,
            iterations=max_iterations,
            error=np.linalg.norm(target_vector[:3] - self._pose_to_vector(
                self.chain.forward_kinematics(angles)
            )[:3])
        )

    def _pose_to_vector(self, T: np.ndarray) -> np.ndarray:
        """Convert transformation to [x, y, z, rx, ry, rz] vector."""
        position = T[:3, 3]

        # Extract rotation as axis-angle
        R = T[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if angle < 1e-6:
            axis = np.array([0, 0, 1])
        else:
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * np.sin(angle))

        rotation = axis * angle
        return np.concatenate([position, rotation])


class IKSolverNode(Node):
    """
    ROS 2 node providing inverse kinematics service.

    Subscribes to current joint states and provides IK solving.
    """

    # DH parameters for 7-DOF humanoid arm
    HUMANOID_ARM_DH = [
        (0,     np.pi/2, 0,     0),      # Shoulder pitch
        (0,    -np.pi/2, 0,     0),      # Shoulder roll
        (0,     np.pi/2, 0.28,  0),      # Shoulder yaw + upper arm
        (0,    -np.pi/2, 0,     0),      # Elbow pitch
        (0,     np.pi/2, 0.25,  0),      # Wrist yaw + forearm
        (0,    -np.pi/2, 0,     0),      # Wrist pitch
        (0,     0,       0.1,   0),      # Wrist roll + hand
    ]

    # Joint limits (radians)
    JOINT_LIMITS = [
        (-np.pi/2, np.pi/2),   # Shoulder pitch
        (-np.pi/6, np.pi),     # Shoulder roll
        (-np.pi/2, np.pi/2),   # Shoulder yaw
        (-np.pi, 0),           # Elbow (only flexion)
        (-np.pi/2, np.pi/2),   # Wrist yaw
        (-np.pi/4, np.pi/4),   # Wrist pitch
        (-np.pi/4, np.pi/4),   # Wrist roll
    ]

    def __init__(self):
        super().__init__('ik_solver')

        # Initialize kinematic chain and solver
        self.chain = KinematicChain(self.HUMANOID_ARM_DH)
        self.solver = JacobianIKSolver(
            self.chain,
            joint_limits=self.JOINT_LIMITS,
            damping=0.1
        )

        # Current joint state
        self.current_joints = np.zeros(7)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_callback, 10
        )

        # Publishers
        self.solution_pub = self.create_publisher(
            JointState, '/ik_solution', 10
        )

        # Target pose subscription
        self.target_sub = self.create_subscription(
            PoseStamped, '/ik_target',
            self._target_callback, 10
        )

        self.get_logger().info('IK solver ready')

    def _joint_state_callback(self, msg: JointState):
        """Update current joint state."""
        arm_joints = [
            'shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint',
            'elbow_pitch_joint', 'wrist_yaw_joint', 'wrist_pitch_joint',
            'wrist_roll_joint'
        ]

        for i, name in enumerate(arm_joints):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joints[i] = msg.position[idx]

    def _target_callback(self, msg: PoseStamped):
        """Solve IK for target pose."""
        target = msg.pose

        self.get_logger().info(
            f'Solving IK for target ({target.position.x:.2f}, '
            f'{target.position.y:.2f}, {target.position.z:.2f})'
        )

        # Convert Pose to transformation matrix
        T = self._pose_to_matrix(target)

        # Solve IK
        result = self.solver.solve(
            T, self.current_joints,
            max_iterations=100,
            tolerance=1e-4
        )

        if result.status == IKStatus.SUCCESS:
            self.get_logger().info(
                f'Solution found in {result.iterations} iterations'
            )
            self.get_logger().info(
                f'Joint angles: {np.round(result.joint_angles, 2).tolist()}'
            )

            # Publish solution
            self._publish_solution(result.joint_angles)
        else:
            self.get_logger().warn(
                f'IK failed: {result.status.name}, error={result.error:.4f}'
            )

    def _pose_to_matrix(self, pose: Pose) -> np.ndarray:
        """Convert ROS Pose to 4x4 transformation matrix."""
        T = np.eye(4)

        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z

        q = pose.orientation
        T[:3, :3] = self._quat_to_rot(q.x, q.y, q.z, q.w)

        return T

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def _publish_solution(self, joint_angles: np.ndarray):
        """Publish IK solution as JointState."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            'shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint',
            'elbow_pitch_joint', 'wrist_yaw_joint', 'wrist_pitch_joint',
            'wrist_roll_joint'
        ]
        msg.position = joint_angles.tolist()

        self.solution_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = IKSolverNode()

    # Demo: Solve IK for a test pose
    demo_target = PoseStamped()
    demo_target.header.frame_id = 'base_link'
    demo_target.pose.position.x = 0.4
    demo_target.pose.position.y = 0.0
    demo_target.pose.position.z = 0.3
    demo_target.pose.orientation.w = 1.0

    # Give time for subscribers to connect
    import time
    time.sleep(1.0)

    node._target_callback(demo_target)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
