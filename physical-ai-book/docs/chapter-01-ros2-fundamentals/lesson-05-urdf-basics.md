---
sidebar_position: 5
title: URDF Basics
description: Describing robots with the Unified Robot Description Format
---

# URDF Basics

## Learning Objectives

By the end of this lesson, you will:

1. Understand URDF structure (links and joints)
2. Read and interpret URDF files
3. Visualize a robot model in RViz
4. Understand the book's humanoid URDF model

## What is URDF?

**URDF** (Unified Robot Description Format) is an XML format that describes:

- **Links**: The rigid bodies of a robot (arms, legs, torso)
- **Joints**: Connections between links (revolute, prismatic, fixed)
- **Visual geometry**: How the robot looks
- **Collision geometry**: Simplified shapes for physics
- **Inertial properties**: Mass and moments of inertia

```xml
<robot name="my_robot">
  <link name="base_link">...</link>
  <joint name="joint1">...</joint>
  <link name="link1">...</link>
</robot>
```

## URDF Structure

### The Robot Tag

Every URDF starts with a `<robot>` tag:

```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  <!-- Links and joints go here -->
</robot>
```

### Links

Links are rigid bodies with three properties:

```xml
<link name="base_link">
  <!-- Visual: How it looks -->
  <visual>
    <geometry>
      <box size="0.5 0.5 0.1"/>
    </geometry>
    <material name="blue">
      <color rgba="0.2 0.4 0.8 1.0"/>
    </material>
  </visual>

  <!-- Collision: Simplified shape for physics -->
  <collision>
    <geometry>
      <box size="0.5 0.5 0.1"/>
    </geometry>
  </collision>

  <!-- Inertial: Mass and moments of inertia -->
  <inertial>
    <mass value="10.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0"
             iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

### Geometry Types

| Type | Description | Example |
|------|-------------|---------|
| `box` | Rectangular prism | `<box size="x y z"/>` |
| `cylinder` | Cylinder | `<cylinder radius="r" length="l"/>` |
| `sphere` | Sphere | `<sphere radius="r"/>` |
| `mesh` | 3D mesh file | `<mesh filename="package://pkg/mesh.stl"/>` |

### Joints

Joints connect two links and define how they move:

```xml
<joint name="shoulder_joint" type="revolute">
  <parent link="base_link"/>
  <child link="upper_arm"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
</joint>
```

### Joint Types

| Type | Motion | DOF | Use Case |
|------|--------|-----|----------|
| `revolute` | Rotation with limits | 1 | Elbow, shoulder |
| `continuous` | Unlimited rotation | 1 | Wheel |
| `prismatic` | Linear sliding | 1 | Linear actuator |
| `fixed` | No motion | 0 | Sensor mount |
| `floating` | 6-DOF | 6 | Base (simulation) |

### Origin and Axis

```xml
<origin xyz="x y z" rpy="roll pitch yaw"/>
```

- `xyz`: Position offset from parent link (meters)
- `rpy`: Rotation offset (radians): roll (X), pitch (Y), yaw (Z)

```xml
<axis xyz="0 0 1"/>
```

- Defines the axis of rotation/translation
- `0 0 1` = Z-axis, `0 1 0` = Y-axis, `1 0 0` = X-axis

## Simple Arm Example

Let's build a 2-DOF arm:

```xml
<?xml version="1.0"?>
<robot name="simple_arm">

  <!-- Base (fixed to world) -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0"
               iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Shoulder joint -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.025" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="1"/>
  </joint>

  <!-- Upper arm -->
  <link name="upper_arm">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <origin xyz="0 0 0.15"/>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <origin xyz="0 0 0.15"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.15"/>
      <inertia ixx="0.005" ixy="0" ixz="0"
               iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Elbow joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="30" velocity="1"/>
  </joint>

  <!-- Lower arm -->
  <link name="lower_arm">
    <visual>
      <geometry>
        <box size="0.04 0.04 0.25"/>
      </geometry>
      <origin xyz="0 0 0.125"/>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.04 0.04 0.25"/>
      </geometry>
      <origin xyz="0 0 0.125"/>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.125"/>
      <inertia ixx="0.002" ixy="0" ixz="0"
               iyy="0.002" iyz="0" izz="0.0005"/>
    </inertial>
  </link>

</robot>
```

## Visualizing in RViz

### Step 1: Create a Launch File

Create `launch/display.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('my_first_pkg'),
        'urdf',
        'simple_arm.urdf'
    )

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),

        # Joint State Publisher GUI
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(
                get_package_share_directory('my_first_pkg'),
                'rviz',
                'display.rviz'
            )]
        )
    ])
```

### Step 2: Launch and View

```bash
ros2 launch my_first_pkg display.launch.py
```

In RViz:
1. Add "RobotModel" display
2. Set "Fixed Frame" to `base_link`
3. Use Joint State Publisher GUI to move joints

## The Book's Humanoid URDF

Our 21-DOF humanoid includes:

| Body Part | Joints | DOF |
|-----------|--------|-----|
| Waist | waist_yaw | 1 |
| Right Arm | shoulder (3) + elbow + wrist (3) | 7 |
| Left Arm | shoulder (3) + elbow + wrist (3) | 7 |
| Right Leg | hip (3) + knee + ankle (2) | 6 |
| Left Leg | hip (3) + knee + ankle (2) | 6 |
| **Total** | | **21** |

### Link Hierarchy

```
base_link (pelvis)
├── torso
│   ├── head
│   │   └── camera_link
│   ├── r_upper_arm_pitch
│   │   └── r_upper_arm_roll
│   │       └── r_upper_arm
│   │           └── r_lower_arm
│   │               └── r_wrist_pitch_link
│   │                   └── r_wrist_roll_link
│   │                       └── r_hand
│   └── l_upper_arm_pitch (mirror)
│       └── ...
├── r_hip_pitch_link
│   └── r_hip_roll_link
│       └── r_upper_leg
│           └── r_lower_leg
│               └── r_ankle_pitch_link
│                   └── r_foot
└── l_hip_pitch_link (mirror)
    └── ...
```

### Joint Naming Convention

```
{side}_{body_part}_{motion_type}

Examples:
- r_shoulder_pitch  → Right shoulder, pitch rotation
- l_hip_roll        → Left hip, roll rotation
- r_wrist_yaw       → Right wrist, yaw rotation
```

## XACRO: URDF Macros

For complex robots, **XACRO** adds programming features:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot">

  <!-- Properties (variables) -->
  <xacro:property name="arm_length" value="0.3"/>

  <!-- Macros (functions) -->
  <xacro:macro name="arm" params="side">
    <link name="${side}_arm">
      <visual>
        <geometry>
          <box size="0.05 0.05 ${arm_length}"/>
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <!-- Use macros -->
  <xacro:arm side="left"/>
  <xacro:arm side="right"/>

</robot>
```

Process XACRO to URDF:

```bash
xacro robot.urdf.xacro > robot.urdf
```

## Inertial Properties

Accurate inertia is critical for physics simulation:

```xml
<inertial>
  <mass value="1.0"/>          <!-- kg -->
  <origin xyz="0 0 0.1"/>      <!-- Center of mass -->
  <inertia
    ixx="0.01" ixy="0" ixz="0"
    iyy="0.01" iyz="0"
    izz="0.01"/>               <!-- kg·m² -->
</inertial>
```

### Common Inertia Formulas

| Shape | Ixx, Iyy | Izz |
|-------|----------|-----|
| **Box** (x,y,z) | m(y²+z²)/12, m(x²+z²)/12 | m(x²+y²)/12 |
| **Cylinder** (r,h) | m(3r²+h²)/12 | mr²/2 |
| **Sphere** (r) | 2mr²/5 | 2mr²/5 |

## Summary

You've learned:

- ✅ URDF structure: links, joints, geometry
- ✅ Joint types: revolute, continuous, prismatic, fixed
- ✅ Visual, collision, and inertial properties
- ✅ How to visualize robots in RViz
- ✅ The book's 21-DOF humanoid structure

## Checkpoint

Can you:

1. Identify links and joints in a URDF file?
2. Explain the difference between visual and collision geometry?
3. Describe the joint configuration of our humanoid's arm?

---

**Next**: [Exercises](/docs/chapter-01-ros2-fundamentals/exercises)
