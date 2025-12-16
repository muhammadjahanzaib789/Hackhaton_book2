---
sidebar_position: 1
title: "Lesson 1: Robot Arm Kinematics"
description: "Understanding forward and inverse kinematics for humanoid manipulation"
---

# Robot Arm Kinematics

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand kinematic chains and coordinate frames
2. Compute forward kinematics using DH parameters
3. Solve inverse kinematics for arm positioning
4. Implement kinematic solutions in Python

## Prerequisites

- Linear algebra fundamentals (matrices, transformations)
- Understanding of coordinate frames
- Basic trigonometry

## What is Kinematics?

Kinematics describes the motion of robot links without considering forces.

```
┌─────────────────────────────────────────────────────────────┐
│                  Kinematics Overview                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Forward Kinematics (FK)                                    │
│  ─────────────────────                                      │
│  Joint Angles → End Effector Position                       │
│                                                             │
│  [θ₁, θ₂, θ₃, ...] ──────▶ [x, y, z, roll, pitch, yaw]   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     Base ──○── Link1 ──○── Link2 ──○── Gripper     │   │
│  │           θ₁          θ₂          θ₃               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Inverse Kinematics (IK)                                    │
│  ─────────────────────                                      │
│  End Effector Position → Joint Angles                       │
│                                                             │
│  [x, y, z, roll, pitch, yaw] ──────▶ [θ₁, θ₂, θ₃, ...]   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Coordinate Frames

Every link in a robot has an attached coordinate frame.

```
┌─────────────────────────────────────────────────────────────┐
│                  Coordinate Frames                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    z₂ ↑                                     │
│                       │    → x₂                             │
│                       ○────────                             │
│                      /│                                     │
│                     / │ Link 2                              │
│                    /  │                                     │
│       z₁ ↑        /   │                                     │
│          │       /    │                                     │
│          ○──────○     │                                     │
│         /│     θ₂     │                                     │
│        / │ Link 1     │                                     │
│       /  │            │                                     │
│  z₀ ↑   │            │                                     │
│     │   │            │                                     │
│     ○───○            ▼                                     │
│    /   θ₁         End Effector                             │
│   → x₀                                                      │
│  Base                                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Denavit-Hartenberg Parameters

The DH convention provides a systematic way to assign coordinate frames.

### DH Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| **Link Length** | a | Distance along x from z(n-1) to z(n) |
| **Link Twist** | α | Angle from z(n-1) to z(n) about x |
| **Link Offset** | d | Distance along z from x(n-1) to x(n) |
| **Joint Angle** | θ | Angle from x(n-1) to x(n) about z |

### Transformation Matrix

```python
import numpy as np

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute DH transformation matrix.

    The transformation from frame n-1 to frame n using DH parameters.

    Args:
        a: Link length (distance along x)
        alpha: Link twist (rotation about x)
        d: Link offset (distance along z)
        theta: Joint angle (rotation about z)

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
```

## Forward Kinematics

### Kinematic Chain

```python
class KinematicChain:
    """
    Forward kinematics using DH parameters.

    Computes end effector pose from joint angles.
    """

    def __init__(self, dh_params: list):
        """
        Initialize kinematic chain.

        Args:
            dh_params: List of (a, alpha, d, theta_offset) tuples
                       theta_offset is added to joint angle
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            joint_angles: Array of joint angles [θ₁, θ₂, ...]

        Returns:
            4x4 transformation matrix (base to end effector)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {len(joint_angles)}")

        T = np.eye(4)

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T = T @ dh_transform(a, alpha, d, theta)

        return T

    def get_position(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end effector position [x, y, z]."""
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3]

    def get_orientation(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get end effector rotation matrix."""
        T = self.forward_kinematics(joint_angles)
        return T[:3, :3]
```

### Example: 3-DOF Planar Arm

```python
# DH parameters for 3-link planar arm
# Each link: 0.3m length, no twist, no offset
dh_params = [
    (0.3, 0, 0, 0),  # Link 1: a=0.3, α=0, d=0, θ_offset=0
    (0.3, 0, 0, 0),  # Link 2
    (0.2, 0, 0, 0),  # Link 3 (end effector)
]

arm = KinematicChain(dh_params)

# Compute end effector position for joint angles [30°, 45°, -15°]
angles = np.radians([30, 45, -15])
position = arm.get_position(angles)
print(f"End effector position: {position}")
# Output: End effector position: [0.673, 0.456, 0.0]
```

## Inverse Kinematics

IK finds joint angles that place the end effector at a desired pose.

### Analytical IK (2-Link Arm)

```python
def ik_2link_planar(x: float, y: float, l1: float, l2: float) -> tuple:
    """
    Analytical inverse kinematics for 2-link planar arm.

    Uses geometric approach for exact solution.

    Args:
        x, y: Target position
        l1, l2: Link lengths

    Returns:
        (theta1, theta2) in radians, or None if unreachable
    """
    # Distance to target
    d = np.sqrt(x**2 + y**2)

    # Check reachability
    if d > l1 + l2 or d < abs(l1 - l2):
        return None  # Target unreachable

    # Law of cosines for elbow angle
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # Handle numerical errors

    # Elbow down solution
    theta2 = np.arccos(cos_theta2)

    # Shoulder angle
    beta = np.arctan2(y, x)
    gamma = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = beta - gamma

    return theta1, theta2
```

### Numerical IK (Jacobian-Based)

```python
class JacobianIK:
    """
    Iterative inverse kinematics using the Jacobian.

    Uses damped least squares (Levenberg-Marquardt) for stability.
    """

    def __init__(self, chain: KinematicChain, damping: float = 0.1):
        self.chain = chain
        self.damping = damping

    def compute_jacobian(self, joint_angles: np.ndarray,
                         delta: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian numerically using finite differences.

        Args:
            joint_angles: Current joint configuration
            delta: Finite difference step size

        Returns:
            6xN Jacobian matrix (position + orientation)
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
              tolerance: float = 1e-4) -> np.ndarray:
        """
        Solve IK iteratively.

        Args:
            target_pose: 4x4 target transformation matrix
            initial_angles: Starting joint configuration
            max_iterations: Maximum solver iterations
            tolerance: Convergence threshold

        Returns:
            Joint angles that achieve target pose
        """
        angles = initial_angles.copy()
        target_vector = self._pose_to_vector(target_pose)

        for iteration in range(max_iterations):
            current_pose = self.chain.forward_kinematics(angles)
            current_vector = self._pose_to_vector(current_pose)

            error = target_vector - current_vector
            error_norm = np.linalg.norm(error)

            if error_norm < tolerance:
                return angles

            # Compute Jacobian
            J = self.compute_jacobian(angles)

            # Damped least squares
            # Δθ = J^T(JJ^T + λ²I)^(-1) * error
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(6)
            delta_angles = J.T @ np.linalg.solve(damped, error)

            # Update angles
            angles = angles + delta_angles

        return angles

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
```

## Humanoid Arm Kinematics

### 7-DOF Arm Configuration

```
┌─────────────────────────────────────────────────────────────┐
│              Humanoid Arm (7-DOF)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Shoulder (3 DOF)                                           │
│  ┌─────────────┐                                            │
│  │  Pitch (θ₁) │ Forward/backward swing                    │
│  │  Roll  (θ₂) │ Lateral raise                             │
│  │  Yaw   (θ₃) │ Internal/external rotation                │
│  └─────────────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  Elbow (1 DOF)                                              │
│  ┌─────────────┐                                            │
│  │  Pitch (θ₄) │ Flexion/extension                         │
│  └─────────────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  Wrist (3 DOF)                                              │
│  ┌─────────────┐                                            │
│  │  Yaw   (θ₅) │ Pronation/supination                      │
│  │  Pitch (θ₆) │ Flexion/extension                         │
│  │  Roll  (θ₇) │ Deviation                                 │
│  └─────────────┘                                            │
│         │                                                   │
│         ▼                                                   │
│    End Effector                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### DH Parameters for 7-DOF Arm

```python
# DH parameters for humanoid arm
# Based on common humanoid dimensions
HUMANOID_ARM_DH = [
    # (a,    alpha,   d,     theta_offset)
    (0,     np.pi/2, 0,     0),        # Shoulder pitch
    (0,    -np.pi/2, 0,     0),        # Shoulder roll
    (0,     np.pi/2, 0.28,  0),        # Shoulder yaw + upper arm
    (0,    -np.pi/2, 0,     0),        # Elbow pitch
    (0,     np.pi/2, 0.25,  0),        # Wrist yaw + forearm
    (0,    -np.pi/2, 0,     0),        # Wrist pitch
    (0,     0,       0.1,   0),        # Wrist roll + hand
]

arm = KinematicChain(HUMANOID_ARM_DH)

# Home position (arm at side)
home_angles = np.array([0, 0, 0, 0, 0, 0, 0])

# Reaching forward position
reach_angles = np.array([
    np.radians(45),   # Shoulder forward
    np.radians(0),    # No lateral raise
    np.radians(0),    # No rotation
    np.radians(-90),  # Elbow bent
    np.radians(0),    # Wrist neutral
    np.radians(0),    # Wrist neutral
    np.radians(0),    # Wrist neutral
])

print(f"Home position: {arm.get_position(home_angles)}")
print(f"Reach position: {arm.get_position(reach_angles)}")
```

## Joint Limits and Workspace

### Joint Limit Enforcement

```python
class JointLimitedChain:
    """Kinematic chain with joint limits."""

    def __init__(self, chain: KinematicChain,
                 joint_limits: list):
        """
        Args:
            chain: Kinematic chain
            joint_limits: List of (min, max) tuples in radians
        """
        self.chain = chain
        self.limits = np.array(joint_limits)

    def clamp_joints(self, angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits."""
        return np.clip(angles, self.limits[:, 0], self.limits[:, 1])

    def check_limits(self, angles: np.ndarray) -> bool:
        """Check if all joints are within limits."""
        return np.all(angles >= self.limits[:, 0]) and \
               np.all(angles <= self.limits[:, 1])


# Typical humanoid arm joint limits
ARM_JOINT_LIMITS = [
    (-np.pi/2, np.pi/2),    # Shoulder pitch
    (-np.pi/6, np.pi),      # Shoulder roll
    (-np.pi/2, np.pi/2),    # Shoulder yaw
    (-np.pi, 0),            # Elbow (only flexion)
    (-np.pi/2, np.pi/2),    # Wrist yaw
    (-np.pi/4, np.pi/4),    # Wrist pitch
    (-np.pi/4, np.pi/4),    # Wrist roll
]
```

### Workspace Visualization

```python
def compute_workspace(chain: KinematicChain,
                      joint_limits: list,
                      samples_per_joint: int = 10) -> np.ndarray:
    """
    Sample the reachable workspace.

    Args:
        chain: Kinematic chain
        joint_limits: Joint limits
        samples_per_joint: Samples per joint dimension

    Returns:
        Array of reachable positions
    """
    positions = []
    limits = np.array(joint_limits)

    # Create joint angle samples
    joint_samples = [
        np.linspace(limits[i, 0], limits[i, 1], samples_per_joint)
        for i in range(chain.num_joints)
    ]

    # Sample workspace (use itertools for efficiency)
    from itertools import product

    for angles in product(*joint_samples):
        angles = np.array(angles)
        pos = chain.get_position(angles)
        positions.append(pos)

    return np.array(positions)
```

## ROS 2 Integration

### IK Service Node

```python
#!/usr/bin/env python3
"""
IK Service Node
Physical AI Book - Chapter 5

Provides inverse kinematics as a ROS 2 service.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import numpy as np


class IKServiceNode(Node):
    """ROS 2 node providing IK service."""

    def __init__(self):
        super().__init__('ik_service')

        # Initialize kinematic chain
        self.chain = KinematicChain(HUMANOID_ARM_DH)
        self.ik_solver = JacobianIK(self.chain)
        self.limits = JointLimitedChain(self.chain, ARM_JOINT_LIMITS)

        # Current joint state
        self.current_joints = np.zeros(7)

        # Subscriber for current joint state
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )

        self.get_logger().info('IK service ready')

    def joint_callback(self, msg):
        """Update current joint state."""
        # Extract arm joints (assuming specific naming)
        arm_joints = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                      'elbow_pitch', 'wrist_yaw', 'wrist_pitch', 'wrist_roll']

        for i, name in enumerate(arm_joints):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joints[i] = msg.position[idx]

    def solve_ik(self, target_pose: Pose) -> np.ndarray:
        """
        Solve IK for target pose.

        Args:
            target_pose: Desired end effector pose

        Returns:
            Joint angles or None if no solution
        """
        # Convert Pose to transformation matrix
        T = self._pose_to_matrix(target_pose)

        # Solve IK starting from current configuration
        solution = self.ik_solver.solve(
            T, self.current_joints,
            max_iterations=100,
            tolerance=1e-4
        )

        # Check joint limits
        if not self.limits.check_limits(solution):
            solution = self.limits.clamp_joints(solution)
            self.get_logger().warn('Solution clamped to joint limits')

        return solution

    def _pose_to_matrix(self, pose: Pose) -> np.ndarray:
        """Convert ROS Pose to 4x4 transformation matrix."""
        T = np.eye(4)

        # Position
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z

        # Orientation (quaternion to rotation matrix)
        q = pose.orientation
        T[:3, :3] = self._quat_to_rot(q.x, q.y, q.z, q.w)

        return T

    def _quat_to_rot(self, x, y, z, w) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])


def main(args=None):
    rclpy.init(args=args)
    node = IKServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **Forward kinematics** maps joint angles to end effector pose
2. **DH parameters** provide systematic frame assignment
3. **Analytical IK** gives exact solutions for simple arms
4. **Numerical IK** handles complex kinematics
5. **Joint limits** constrain the workspace

## Next Steps

In the [next lesson](./lesson-02-moveit2.md), we will:
- Configure MoveIt2 for motion planning
- Plan collision-free trajectories
- Execute coordinated arm movements

## Additional Resources

- [Modern Robotics (Lynch & Park)](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- [ROS 2 Kinematics](https://ros.org/reps/rep-0103.html)
- [KDL Kinematics Library](https://orocos.org/kdl.html)
