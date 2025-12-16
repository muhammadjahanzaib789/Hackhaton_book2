---
sidebar_position: 1
title: "Lesson 1: Capstone Project Overview"
description: "Building an integrated physical AI assistant"
---

# Capstone Project Overview

## Project Introduction

Welcome to the capstone project! You will build a complete **Physical AI Assistant** that integrates all concepts from this book:

- ROS 2 robot control
- Computer vision and perception
- Navigation and path planning
- Manipulation and grasping
- LLM-based task understanding
- Voice interaction

## Learning Objectives

By the end of this project, you will:

1. Architect a complete physical AI system
2. Integrate multiple robot subsystems
3. Handle real-world complexity and failures
4. Deploy a production-ready robot application

## Project: Home Assistant Robot

```
┌─────────────────────────────────────────────────────────────┐
│              Home Assistant Robot                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "Hey Robot, bring me a cup of water from the kitchen"      │
│                                                             │
│       ┌─────────────────────────────────────────────────┐  │
│       │              CAPABILITIES                        │  │
│       ├─────────────────────────────────────────────────┤  │
│       │                                                 │  │
│       │  🎤 Voice Command Reception                     │  │
│       │     "Bring me water"                           │  │
│       │                                                 │  │
│       │  🧠 LLM Task Planning                          │  │
│       │     Decompose into subtasks                    │  │
│       │                                                 │  │
│       │  🗺️ Navigation                                 │  │
│       │     Navigate to kitchen                        │  │
│       │                                                 │  │
│       │  👁️ Object Detection                          │  │
│       │     Find cup and water dispenser               │  │
│       │                                                 │  │
│       │  🤖 Manipulation                               │  │
│       │     Pick cup, fill with water                  │  │
│       │                                                 │  │
│       │  🔊 Voice Feedback                             │  │
│       │     "Here's your water"                        │  │
│       │                                                 │  │
│       └─────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              System Architecture                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 COORDINATOR NODE                     │   │
│  │         (State Machine & Task Orchestration)        │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                │                │                   │
│       ▼                ▼                ▼                   │
│  ┌─────────┐    ┌─────────────┐   ┌─────────────┐         │
│  │  Voice  │    │     LLM     │   │   Safety    │         │
│  │Interface│    │  Planner    │   │  Monitor    │         │
│  └─────────┘    └─────────────┘   └─────────────┘         │
│       │                │                │                   │
│       ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 ACTION LAYER                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │Navigation│ │Perception│ │Manipulate│            │   │
│  │  │ (Nav2)   │ │ (Vision) │ │(MoveIt2) │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                │                │                   │
│       ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 HARDWARE LAYER                       │   │
│  │  [Motors] [Cameras] [Lidar] [Gripper] [Mic/Speaker]│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Voice Interface
- Wake word detection
- Speech-to-text (Whisper)
- Text-to-speech (Piper)
- Conversational context

### 2. LLM Task Planner
- Command understanding
- Task decomposition
- Replanning on failure
- Multi-step execution

### 3. Navigation System
- SLAM and localization
- Path planning
- Obstacle avoidance
- Goal tracking

### 4. Perception Pipeline
- Object detection
- Scene understanding
- Target tracking
- Spatial reasoning

### 5. Manipulation System
- Grasp planning
- Motion planning
- Force control
- Object handling

### 6. Coordinator
- State machine
- Task orchestration
- Error handling
- Safety monitoring

## Project Phases

```
┌─────────────────────────────────────────────────────────────┐
│              Implementation Phases                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Infrastructure (Lesson 2)                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                          │
│  ☐ Set up ROS 2 workspace                                  │
│  ☐ Create package structure                                │
│  ☐ Define message types                                    │
│  ☐ Configure simulation                                    │
│                                                             │
│  Phase 2: Core Components (Lesson 3)                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                           │
│  ☐ Implement voice interface                               │
│  ☐ Integrate LLM planner                                   │
│  ☐ Configure Nav2                                          │
│  ☐ Set up perception pipeline                              │
│  ☐ Configure MoveIt2                                       │
│                                                             │
│  Phase 3: Integration (Lesson 4)                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                             │
│  ☐ Build coordinator node                                  │
│  ☐ Implement state machine                                 │
│  ☐ Create action clients                                   │
│  ☐ Add safety monitoring                                   │
│                                                             │
│  Phase 4: Testing & Refinement (Lesson 5)                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                       │
│  ☐ Unit testing                                            │
│  ☐ Integration testing                                     │
│  ☐ Simulation testing                                      │
│  ☐ Performance optimization                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ROS 2 Package Structure

```
home_assistant_robot/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── home_assistant_robot/
│   ├── __init__.py
│   ├── coordinator.py          # Main coordinator
│   ├── voice_interface.py      # Voice I/O
│   ├── llm_planner.py          # Task planning
│   ├── perception_node.py      # Object detection
│   ├── manipulation_node.py    # Arm control
│   └── safety_monitor.py       # Safety checks
├── msg/
│   ├── Task.msg
│   ├── Action.msg
│   └── SystemStatus.msg
├── srv/
│   ├── PlanTask.srv
│   └── ExecuteAction.srv
├── action/
│   ├── ExecuteTask.action
│   └── PickPlace.action
├── config/
│   ├── robot_params.yaml
│   ├── nav2_params.yaml
│   └── perception_params.yaml
├── launch/
│   ├── full_system.launch.py
│   ├── simulation.launch.py
│   └── hardware.launch.py
└── test/
    ├── test_coordinator.py
    ├── test_planner.py
    └── test_integration.py
```

## Success Criteria

Your project should demonstrate:

| Criterion | Description |
|-----------|-------------|
| **Voice Control** | Accept and understand voice commands |
| **Task Planning** | Decompose commands into executable steps |
| **Navigation** | Navigate to specified locations |
| **Object Detection** | Find and locate target objects |
| **Manipulation** | Pick up and place objects |
| **Error Handling** | Recover from failures gracefully |
| **Safety** | Never endanger humans or property |
| **Feedback** | Provide status updates via voice |

## Getting Started

### Prerequisites

```bash
# Install dependencies
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-moveit
pip install openai-whisper torch torchvision

# Clone starter code (if available)
cd ~/ros2_ws/src
git clone https://github.com/physical-ai-book/capstone-starter.git
cd ..
colcon build
```

### Initial Setup

```python
# Launch simulation environment
ros2 launch home_assistant_robot simulation.launch.py

# In another terminal, run the coordinator
ros2 run home_assistant_robot coordinator

# Test with a voice command
ros2 topic pub /voice/command std_msgs/String "data: 'go to the kitchen'"
```

## Summary

In this capstone project, you will:

1. **Design** a complete physical AI system architecture
2. **Implement** each component using skills from previous chapters
3. **Integrate** all components into a cohesive system
4. **Test** thoroughly in simulation and potentially hardware
5. **Document** your design decisions and learnings

## Next Steps

Continue to [Lesson 2](./lesson-02-implementation.md) to begin implementation:
- Set up the project structure
- Implement core nodes
- Create the coordinator

## Project Resources

- [Starter Code Repository](https://github.com/physical-ai-book/capstone)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Nav2 Tutorials](https://navigation.ros.org/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/)
