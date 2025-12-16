---
sidebar_position: 2
title: "Learning Paths"
description: "Recommended paths through the Physical AI Book based on your goals"
---

# Learning Paths

Choose a learning path based on your goals and background. Each path provides a focused journey through the book's content.

## Path 1: Complete Beginner

**For:** Those new to robotics and ROS 2

**Duration:** 8-10 weeks (part-time study)

**Prerequisites:** Basic Python programming

```
Week 1-2: ROS 2 Fundamentals
├── Chapter 1: All lessons
└── Focus: Understand nodes, topics, services

Week 3-4: Simulation
├── Chapter 2: All lessons
├── Build: First simulated robot
└── Focus: URDF, Gazebo basics

Week 5-6: Perception & Navigation
├── Chapter 3: Lessons 1-2
├── Chapter 4: Lessons 1-2
└── Focus: Basic sensors and SLAM

Week 7-8: Integration
├── Chapter 6: Lesson 1 (LLM basics)
├── Chapter 8: Lessons 1-2
└── Build: Simple assistant robot

Week 9-10: Capstone
├── Chapter 8: Complete capstone
└── Deploy: Working robot demo
```

### Key Milestones

1. **Week 2:** Publish your first custom message
2. **Week 4:** Simulate a robot in Gazebo
3. **Week 6:** Navigate autonomously in simulation
4. **Week 8:** Voice-controlled robot
5. **Week 10:** Complete home assistant demo

---

## Path 2: ROS 1 Developer Transitioning to ROS 2

**For:** Experienced ROS 1 developers

**Duration:** 3-4 weeks

**Prerequisites:** ROS 1 experience

```
Week 1: ROS 2 Differences
├── Chapter 1: Lessons 3-5 (skip basics)
├── Focus: ROS 2 vs ROS 1 differences
│   - DDS middleware
│   - Action servers
│   - Lifecycle nodes
└── Reference: Migration guide

Week 2: Modern Tools
├── Chapter 2: Lessons 3-5 (new Gazebo)
├── Chapter 4: Lesson 2 (Nav2)
└── Focus: New tooling and APIs

Week 3: AI Integration
├── Chapter 6: All lessons
├── Chapter 7: Lesson 1
└── Focus: LLM and VLA integration

Week 4: Advanced Topics
├── Chapter 5: Lessons 2-3 (MoveIt2)
├── Chapter 8: Integration lessons
└── Build: Advanced robot system
```

### Key Differences to Master

| Concept | ROS 1 | ROS 2 |
|---------|-------|-------|
| Middleware | Custom | DDS |
| Python API | rospy | rclpy |
| C++ API | roscpp | rclcpp |
| Launch | XML | Python/XML/YAML |
| Params | Master | Node-local |
| Navigation | move_base | Nav2 |
| Manipulation | MoveIt | MoveIt2 |

---

## Path 3: AI/ML Specialist Adding Robotics

**For:** ML engineers wanting to deploy models on robots

**Duration:** 4-5 weeks

**Prerequisites:** Python, PyTorch/TensorFlow, ML fundamentals

```
Week 1: Robotics Foundations
├── Chapter 1: Lessons 1, 3, 4 (essentials)
├── Chapter 2: Lesson 4 (sensors)
└── Focus: ROS 2 communication

Week 2: Perception Pipeline
├── Chapter 3: All lessons
├── Build: Custom detector node
└── Focus: Camera to model integration

Week 3: LLM Integration
├── Chapter 6: All lessons (deep dive)
├── Build: Voice-controlled planner
└── Focus: Action schemas, safety

Week 4: VLA Models
├── Chapter 7: All lessons
├── Build: End-to-end VLA system
└── Focus: Training and deployment

Week 5: Production Deployment
├── Chapter 8: Lessons 3-4
├── Focus: Testing, optimization
└── Deploy: Complete AI robot
```

### ML Integration Points

```
Perception (Chapter 3)
├── Image preprocessing
├── Model inference
├── Post-processing
└── ROS 2 publishing

LLM Planning (Chapter 6)
├── Prompt engineering
├── Action schemas
├── Safety validation
└── Error recovery

VLA Control (Chapter 7)
├── Vision encoding
├── Language embedding
├── Action decoding
└── Real-time inference
```

---

## Path 4: Hardware Engineer Going Software

**For:** Mechanical/electrical engineers adding software skills

**Duration:** 6-8 weeks

**Prerequisites:** Basic programming, robot hardware familiarity

```
Week 1-2: Software Fundamentals
├── Chapter 1: All lessons
├── Chapter 2: Lesson 2 (URDF)
└── Focus: Describing your robot

Week 3-4: Simulation
├── Chapter 2: All lessons
├── Build: Digital twin of hardware
└── Focus: Physics, sensors

Week 5-6: Control & Navigation
├── Chapter 4: All lessons
├── Chapter 5: All lessons
└── Focus: Controllers, planning

Week 7-8: Integration
├── Chapter 8: All lessons
└── Build: Complete software stack
```

### Hardware to Software Bridge

```
Physical Robot
├── Actuators → Joint controllers
├── Sensors → ROS 2 drivers
├── Structure → URDF model
└── Electronics → ros2_control

Software Stack
├── Low-level: Controllers
├── Mid-level: Planning
├── High-level: Task management
└── Top-level: AI integration
```

---

## Path 5: Quick Start (Weekend Project)

**For:** Experienced developers wanting fast results

**Duration:** 1 weekend

**Prerequisites:** Programming experience, Docker

```
Saturday Morning: Setup
├── Chapter 1: Lesson 2 (installation)
├── Docker-based quick start
└── Run: Demo simulations

Saturday Afternoon: Core Concepts
├── Chapter 1: Lessons 3-4 (skim)
├── Chapter 4: Lesson 2 (Nav2)
└── Run: Navigation demo

Sunday Morning: AI Integration
├── Chapter 6: Lesson 1 (LLM)
├── Set up: Ollama local LLM
└── Build: Voice command interface

Sunday Afternoon: Demo
├── Chapter 8: Lesson 1 (overview)
├── Run: Complete demo
└── Customize: Your commands
```

### Quick Start Commands

```bash
# Clone and run
git clone https://github.com/physical-ai-book/examples
cd examples
docker-compose up simulation

# In another terminal
docker exec -it robot bash
ros2 launch home_assistant_robot demo.launch.py
```

---

## Path 6: Research-Focused

**For:** Graduate students and researchers

**Duration:** Self-paced (reference guide)

**Prerequisites:** Strong programming, ML knowledge

```
Core Reading (1 week)
├── Chapter 1: Architecture overview
├── Chapter 2: Simulation tools
└── Focus: Experiment infrastructure

Research Areas:
├── Perception Research
│   └── Chapter 3: Deep dive
├── Navigation Research
│   └── Chapter 4: All + papers
├── Manipulation Research
│   └── Chapter 5: All + papers
├── LLM Research
│   └── Chapter 6: All + papers
└── VLA Research
    └── Chapter 7: All + papers
```

### Research Extensions

| Topic | Chapter | Research Directions |
|-------|---------|---------------------|
| Perception | 3 | Novel architectures, few-shot |
| Navigation | 4 | Learning-based planners |
| Manipulation | 5 | Contact-rich tasks |
| LLM | 6 | Grounding, reasoning |
| VLA | 7 | Sample efficiency, sim2real |

---

## Path Selection Guide

Answer these questions to find your path:

1. **What's your ROS experience?**
   - None → Path 1 (Beginner)
   - ROS 1 → Path 2 (Transition)
   - ROS 2 → Path 3, 4, or 5

2. **What's your primary skill?**
   - ML/AI → Path 3 (AI Specialist)
   - Hardware → Path 4 (HW Engineer)
   - Software → Path 1 or 5

3. **How much time do you have?**
   - Weekend → Path 5 (Quick Start)
   - A few weeks → Path 2 or 3
   - A few months → Path 1 or 4
   - Self-paced → Path 6 (Research)

4. **What's your goal?**
   - Learn fundamentals → Path 1
   - Build quickly → Path 5
   - Research → Path 6
   - Career transition → Path 1 or 4

## Progress Tracking

Use this checklist to track your progress through any path:

### Fundamentals
- [ ] Create a ROS 2 workspace
- [ ] Write a publisher and subscriber
- [ ] Create a custom message
- [ ] Write a service server
- [ ] Implement an action server

### Simulation
- [ ] Create a URDF robot model
- [ ] Spawn robot in Gazebo
- [ ] Add sensors to simulation
- [ ] Configure physics properties
- [ ] Set up ROS-Gazebo bridge

### Perception
- [ ] Process camera images
- [ ] Run object detection
- [ ] Publish detections as ROS messages
- [ ] Integrate with depth data

### Navigation
- [ ] Generate a map with SLAM
- [ ] Localize robot with AMCL
- [ ] Configure Nav2
- [ ] Execute autonomous navigation

### Manipulation
- [ ] Configure MoveIt2
- [ ] Plan arm trajectories
- [ ] Execute pick and place

### AI Integration
- [ ] Set up local LLM
- [ ] Create action schemas
- [ ] Implement voice interface
- [ ] Build task planner

### Capstone
- [ ] Integrate all subsystems
- [ ] Write integration tests
- [ ] Deploy complete system
- [ ] Demo to others
