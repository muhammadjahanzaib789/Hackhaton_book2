---
sidebar_position: 2
title: "Lesson 2: URDF Deep Dive"
description: "Advanced URDF concepts for humanoid robot description"
---

# URDF Deep Dive

## Learning Objectives

By the end of this lesson, you will be able to:

1. Configure complex joint types and limits
2. Calculate and specify inertial properties
3. Create collision-safe robot models
4. Debug common URDF issues

## Prerequisites

- Completed Lesson 1 (Gazebo Intro)
- Basic understanding of URDF from Chapter 1
- Familiarity with 3D coordinate systems

## Introduction

While Chapter 1 introduced URDF basics, humanoid robots require careful attention to:

- **Joint Limits**: Prevent impossible configurations
- **Inertial Properties**: Enable realistic dynamics
- **Collision Geometry**: Efficient and accurate physics
- **Visual Fidelity**: Debug-friendly visualization

## Joint Types for Humanoids

Humanoid robots use several joint types:

```
┌─────────────────────────────────────────────────────────────┐
│                     Joint Types                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Revolute (Most Common)          Continuous                 │
│  ┌─────┐                         ┌─────┐                   │
│  │     │◀──rotation──▶│         │     │◀──∞ rotation      │
│  └─────┘ with limits             └─────┘ no limits         │
│                                                             │
│  Prismatic                       Fixed                      │
│  ┌─────┐                         ┌─────┐                   │
│  │     │◀──slide──▶│            │     │ no movement       │
│  └─────┘ linear                  └─────┘ rigid connection  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Revolute Joint Configuration

```xml
<joint name="left_knee_pitch" type="revolute">
  <!-- Parent and child links -->
  <parent link="left_thigh"/>
  <child link="left_shin"/>

  <!-- Joint origin relative to parent -->
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>

  <!-- Rotation axis (in joint frame) -->
  <axis xyz="0 1 0"/>

  <!-- Joint limits -->
  <limit
    lower="0.0"
    upper="2.5"
    effort="100.0"
    velocity="5.0"
  />

  <!-- Dynamics (friction, damping) -->
  <dynamics damping="0.1" friction="0.0"/>

  <!-- Safety controller (optional) -->
  <safety_controller
    soft_lower_limit="0.05"
    soft_upper_limit="2.45"
    k_position="100"
    k_velocity="10"
  />
</joint>
```

### Understanding Joint Limits

| Parameter | Description | Humanoid Typical |
|-----------|-------------|------------------|
| `lower` | Minimum angle (rad) | Anatomically limited |
| `upper` | Maximum angle (rad) | Anatomically limited |
| `effort` | Maximum torque (N·m) | 50-200 for legs |
| `velocity` | Maximum speed (rad/s) | 3-10 |

:::tip Humanoid Joint Limits
Research human joint ranges of motion (ROM) when setting limits. For example:
- Knee flexion: 0° to ~140°
- Hip flexion: ~-15° to ~120°
- Shoulder rotation: varies by axis
:::

## Inertial Properties

Correct inertial properties are **critical** for realistic simulation.

### Inertia Tensor

```
┌─────────────────────────────────────────────────────────────┐
│                    Inertia Tensor                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│       ┌                     ┐                               │
│       │ Ixx   Ixy   Ixz    │                               │
│   I = │ Ixy   Iyy   Iyz    │                               │
│       │ Ixz   Iyz   Izz    │                               │
│       └                     ┘                               │
│                                                             │
│   Diagonal (Ixx, Iyy, Izz): Resistance to rotation         │
│   Off-diagonal: Coupling between axes                      │
│                                                             │
│   For symmetric objects, off-diagonal = 0                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Calculating Inertia for Common Shapes

**Solid Cylinder** (limbs):
```
Ixx = Iyy = (1/12) * m * (3r² + h²)
Izz = (1/2) * m * r²
```

**Solid Box** (torso):
```
Ixx = (1/12) * m * (y² + z²)
Iyy = (1/12) * m * (x² + z²)
Izz = (1/12) * m * (x² + y²)
```

### URDF Inertial Element

```xml
<link name="left_thigh">
  <inertial>
    <!-- Center of mass relative to link origin -->
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>

    <!-- Mass in kg -->
    <mass value="3.5"/>

    <!-- Inertia tensor (symmetric, so only 6 values) -->
    <inertia
      ixx="0.05" ixy="0.0" ixz="0.0"
      iyy="0.05" iyz="0.0"
      izz="0.01"
    />
  </inertial>

  <!-- Visual and collision... -->
</link>
```

### Inertia Calculation Script

```python
#!/usr/bin/env python3
"""
Inertia Calculator for URDF
Physical AI Book - Chapter 2

Calculates inertia tensor for common shapes.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InertiaResult:
    """Result of inertia calculation."""
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    def to_urdf(self, indent: int = 6) -> str:
        """Generate URDF inertia element."""
        spaces = " " * indent
        return (
            f'{spaces}<inertia\n'
            f'{spaces}  ixx="{self.ixx:.6f}" ixy="{self.ixy:.6f}" '
            f'ixz="{self.ixz:.6f}"\n'
            f'{spaces}  iyy="{self.iyy:.6f}" iyz="{self.iyz:.6f}"\n'
            f'{spaces}  izz="{self.izz:.6f}"\n'
            f'{spaces}/>'
        )


def cylinder_inertia(mass: float, radius: float, height: float) -> InertiaResult:
    """
    Calculate inertia for a solid cylinder aligned with Z axis.

    Args:
        mass: Mass in kg
        radius: Radius in meters
        height: Height in meters

    Returns:
        InertiaResult with calculated values
    """
    ixx = iyy = (1/12) * mass * (3 * radius**2 + height**2)
    izz = (1/2) * mass * radius**2
    return InertiaResult(ixx=ixx, iyy=iyy, izz=izz)


def box_inertia(mass: float, x: float, y: float, z: float) -> InertiaResult:
    """
    Calculate inertia for a solid box.

    Args:
        mass: Mass in kg
        x, y, z: Dimensions in meters

    Returns:
        InertiaResult with calculated values
    """
    ixx = (1/12) * mass * (y**2 + z**2)
    iyy = (1/12) * mass * (x**2 + z**2)
    izz = (1/12) * mass * (x**2 + y**2)
    return InertiaResult(ixx=ixx, iyy=iyy, izz=izz)


def sphere_inertia(mass: float, radius: float) -> InertiaResult:
    """
    Calculate inertia for a solid sphere.

    Args:
        mass: Mass in kg
        radius: Radius in meters

    Returns:
        InertiaResult with calculated values
    """
    i = (2/5) * mass * radius**2
    return InertiaResult(ixx=i, iyy=i, izz=i)


# Example: Calculate for humanoid thigh
if __name__ == "__main__":
    # Thigh as cylinder: 0.4m long, 0.06m radius, 3.5kg
    thigh = cylinder_inertia(mass=3.5, radius=0.06, height=0.4)
    print("Thigh inertia:")
    print(thigh.to_urdf())

    # Torso as box: 0.3m x 0.2m x 0.5m, 10kg
    torso = box_inertia(mass=10, x=0.3, y=0.2, z=0.5)
    print("\nTorso inertia:")
    print(torso.to_urdf())
```

### Expected Output

```
Thigh inertia:
      <inertia
        ixx="0.050033" ixy="0.000000" ixz="0.000000"
        iyy="0.050033" iyz="0.000000"
        izz="0.006300"
      />

Torso inertia:
      <inertia
        ixx="0.241667" ixy="0.000000" ixz="0.000000"
        iyy="0.283333" iyz="0.000000"
        izz="0.108333"
      />
```

:::warning Inertia Matters
Incorrect inertia values cause:
- Unrealistic movement
- Simulation instability
- Controller tuning issues
- Failed sim-to-real transfer
:::

## Collision Geometry

Collision geometry determines physics interactions.

### Visual vs. Collision

```
┌─────────────────────────────────────────────────────────────┐
│             Visual vs. Collision Geometry                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Visual (Rendering)           Collision (Physics)          │
│  ┌─────────────────┐          ┌─────────────────┐         │
│  │  High-detail    │          │  Simplified     │         │
│  │  mesh/DAE       │          │  primitives     │         │
│  │  1000+ polys    │          │  or convex hull │         │
│  └─────────────────┘          └─────────────────┘         │
│                                                             │
│  Purpose:                     Purpose:                     │
│  - User visualization         - Physics simulation         │
│  - Debug inspection           - Contact detection          │
│  - Screenshots/video          - Performance                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Efficient Collision Design

```xml
<link name="left_forearm">
  <!-- High-detail visual -->
  <visual>
    <geometry>
      <mesh filename="package://physical_ai_book/meshes/forearm.dae"/>
    </geometry>
    <material name="robot_gray"/>
  </visual>

  <!-- Simplified collision: cylinder approximation -->
  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.04" length="0.3"/>
    </geometry>
  </collision>
</link>
```

### Collision Primitives

| Primitive | Use Case | Performance |
|-----------|----------|-------------|
| **Box** | Rectangular parts | Fastest |
| **Cylinder** | Limbs | Fast |
| **Sphere** | Hands, joints | Fast |
| **Mesh** | Complex shapes | Slow |

:::tip Performance Rule
Use the simplest collision geometry that captures the essential shape. A humanoid with primitive collisions can run 10x faster than one with mesh collisions.
:::

## URDF Validation and Debugging

### Check URDF Syntax

```bash
# Install URDF tools
sudo apt install liburdfdom-tools

# Check URDF file
check_urdf humanoid.urdf

# Expected output (success):
robot name is: humanoid
---------- Successfully Parsed XML ---------------
root Link: base_link has 1 child(ren)
    child(1):  torso
        child(1):  head
        child(2):  left_upper_arm
        ...
```

### Visualize URDF

```bash
# Generate visualization graph
urdf_to_graphviz humanoid.urdf

# This creates humanoid.gv and humanoid.pdf
# View the link/joint tree structure
```

### Common URDF Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Link has no inertia" | Missing `<inertial>` | Add inertial to all non-fixed links |
| "Joint has no limits" | Missing limits on revolute | Add `<limit>` element |
| "Self-collision" | Links intersect | Adjust collision geometry |
| "Unstable simulation" | Bad inertia values | Recalculate inertias |

### URDF Debugging Launch File

```python
#!/usr/bin/env python3
"""
URDF Visualization Launch File
Physical AI Book - Chapter 2

Launches RViz2 with URDF visualization for debugging.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    urdf_path = '/path/to/physical-ai-book/static/models/humanoid/humanoid.urdf'

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['cat ', urdf_path]),
                    value_type=str
                )
            }]
        ),

        # Joint state publisher GUI
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', '/path/to/humanoid.rviz']
        ),
    ])
```

## Complete Humanoid Link Example

Here's a complete, well-configured link for the humanoid's upper arm:

```xml
<link name="left_upper_arm">
  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <mass value="2.0"/>
    <inertia
      ixx="0.0135" ixy="0.0" ixz="0.0"
      iyy="0.0135" iyz="0.0"
      izz="0.002"
    />
  </inertial>

  <!-- Visual representation -->
  <visual>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
    <material name="arm_material">
      <color rgba="0.7 0.7 0.7 1.0"/>
    </material>
  </visual>

  <!-- Collision geometry (simplified) -->
  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.055" length="0.32"/>
    </geometry>
  </collision>
</link>

<!-- Joint connecting to torso -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.0 0.2 0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="2.0" effort="100" velocity="5"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>
```

## Summary

Key takeaways from this lesson:

1. **Joint types** must match the intended motion
2. **Joint limits** should reflect realistic constraints
3. **Inertial properties** are critical for dynamics
4. **Collision geometry** affects both accuracy and performance
5. **Validation tools** catch errors before simulation

## Next Steps

In the [next lesson](./lesson-03-physics-config.md), we will:
- Configure physics engine parameters
- Tune simulation for stability
- Optimize performance

## Additional Resources

- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [SDF Format](http://sdformat.org/)
- [Inertia Calculator Tools](https://www.meshlab.net/)
