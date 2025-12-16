---
sidebar_position: 1
title: Introduction to ROS 2
description: Understanding ROS 2 as the robotic nervous system
---

# Introduction to ROS 2

## Learning Objectives

By the end of this lesson, you will:

1. Understand what ROS 2 is and why it matters for robotics
2. Know the difference between ROS 1 and ROS 2
3. Understand the core concepts: nodes, topics, services, and actions
4. Recognize how ROS 2 serves as the "nervous system" for humanoid robots

## What is ROS 2?

**ROS 2** (Robot Operating System 2) is not actually an operating system. It's a **middleware framework** that provides:

- **Communication infrastructure** between software components
- **Hardware abstraction** for sensors and actuators
- **Standard interfaces** for common robotics tasks
- **Tools** for visualization, debugging, and simulation

Think of ROS 2 as the **nervous system** of a robot. Just as your nervous system connects your brain to your muscles and senses, ROS 2 connects AI algorithms to physical hardware.

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Robot's Brain                      │
│  (LLMs, Planners, Controllers, Vision Systems)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROS 2 Middleware                        │
│  • Message passing                                          │
│  • Service calls                                            │
│  • Action management                                        │
│  • Parameter handling                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Your Robot's Body                       │
│  (Motors, Cameras, LIDAR, IMU, Speakers)                    │
└─────────────────────────────────────────────────────────────┘
```

## Why ROS 2 for Physical AI?

### 1. Real-Time Capabilities

Physical AI requires **deterministic timing**. When you command a robot arm to move, it must respond within milliseconds. ROS 2 supports real-time communication through DDS (Data Distribution Service).

### 2. Multi-Language Support

Your AI might be in Python (PyTorch, TensorFlow), but your motor controllers might need C++. ROS 2 allows seamless communication between them.

### 3. Massive Ecosystem

Thousands of packages for:
- Navigation (Nav2)
- Manipulation (MoveIt 2)
- Perception (Isaac ROS)
- Simulation (Gazebo)

### 4. Industry Standard

Companies like Boston Dynamics, Clearpath, and ABB use ROS 2. Skills transfer directly to industry.

## ROS 2 vs ROS 1

| Aspect | ROS 1 | ROS 2 |
|--------|-------|-------|
| **Communication** | Custom (roscore) | DDS standard |
| **Real-time** | Limited | Full support |
| **Security** | None | Built-in DDS security |
| **Multi-robot** | Complex | Native support |
| **Platforms** | Linux only | Linux, Windows, macOS |
| **Lifecycle** | None | Managed node lifecycle |

**Key insight**: ROS 1 is deprecated. All new development should use ROS 2.

## Core Concepts

### Nodes

A **node** is a single-purpose process. Examples:
- Camera node: publishes images
- Detection node: finds objects in images
- Motor node: controls joint positions

```python
# A simple ROS 2 node structure
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node started!')

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Topics

**Topics** are named buses for continuous data streams. Publishers send data; subscribers receive it.

```
┌──────────────┐                    ┌──────────────┐
│ Camera Node  │ ──/camera/image──▶ │ Vision Node  │
│ (Publisher)  │                    │ (Subscriber) │
└──────────────┘                    └──────────────┘
```

### Services

**Services** are request/response interactions. Client asks, server responds once.

```
┌──────────────┐    Request          ┌──────────────┐
│ Planner Node │ ──/get_plan──▶      │ Path Server  │
│   (Client)   │ ◀── Response ───    │   (Server)   │
└──────────────┘                     └──────────────┘
```

### Actions

**Actions** are long-running tasks with feedback. Essential for physical actions like "move arm to position."

```
┌──────────────┐    Goal             ┌──────────────┐
│ Task Manager │ ──/move_arm──▶      │  Arm Server  │
│   (Client)   │ ◀── Feedback ────   │  (Server)    │
│              │ ◀── Result ──────   │              │
└──────────────┘                     └──────────────┘
```

## The Humanoid Robot Connection

In our humanoid robot, ROS 2 connects:

| Component | ROS 2 Role |
|-----------|------------|
| Voice Input | Topic: `/voice/transcript` |
| LLM Planner | Service: `/llm/decompose_task` |
| Navigation | Action: `/navigate_to_pose` |
| Vision | Topic: `/vision/detections` |
| Manipulation | Action: `/grasp_object` |

Every component communicates through ROS 2 interfaces, creating a modular, testable system.

## Quality of Service (QoS)

ROS 2 allows fine-grained control over message delivery:

- **Reliable**: Guaranteed delivery (use for commands)
- **Best Effort**: Fast but may drop (use for sensor streams)
- **Transient Local**: Late subscribers get last message (use for maps)

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# For commands that must arrive
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    depth=10
)

# For high-frequency sensor data
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=1
)
```

## Summary

- ROS 2 is the **middleware** connecting AI to physical hardware
- It provides **nodes**, **topics**, **services**, and **actions**
- It's the **industry standard** for professional robotics
- Our humanoid uses ROS 2 for **all component communication**

## Next Steps

In the next lesson, we'll install ROS 2 and run our first nodes.

## Checkpoint

Can you answer these questions?

1. What is the difference between a topic and a service?
2. Why would you use an action instead of a service for arm movement?
3. What QoS setting would you use for camera images vs motor commands?

---

**Next**: [Lesson 2: Installation](/docs/chapter-01-ros2-fundamentals/lesson-02-installation)
