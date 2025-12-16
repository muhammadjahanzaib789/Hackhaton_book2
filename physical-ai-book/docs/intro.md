---
sidebar_position: 1
title: Introduction
description: Welcome to Physical AI & Humanoid Robotics
---

# Physical AI & Humanoid Robotics

> **Embodied Intelligence in the Real World**

Welcome to the comprehensive guide for building intelligent humanoid robots. This book will take you from foundational concepts to deploying an autonomous humanoid system that can understand voice commands, reason about tasks, navigate environments, and manipulate objects.

## Core Thesis

> Intelligence reaches its full potential only when it is embodied. Physical AI bridges the digital brain and the physical body, enabling machines to perceive, reason, and act in the real world.

## What You'll Learn

By completing this book, you will be able to:

1. **Build ROS 2 Applications** - Create nodes, topics, services, and actions for robot control
2. **Simulate Humanoid Robots** - Use Gazebo Sim for realistic physics testing
3. **Implement Perception Systems** - Process camera and LIDAR data for scene understanding
4. **Navigate Autonomously** - Use SLAM and Nav2 for robot navigation
5. **Control Robot Arms** - Apply inverse kinematics and motion planning
6. **Integrate LLMs** - Connect language models to robot actions safely
7. **Build VLA Pipelines** - Create end-to-end vision-language-action systems
8. **Complete a Capstone** - Build an autonomous humanoid that fetches objects on command

## Prerequisites

This book assumes:

- **Programming**: Basic Python and/or C++ experience
- **Math**: High school algebra and basic trigonometry
- **System**: Linux system or Docker for development

**No prior robotics experience required.** We start from first principles.

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Chapter 1: ROS 2 Fundamentals                              │
│      ↓                                                      │
│  Chapter 2: Simulation with Gazebo                          │
│      ↓                                                      │
│  Chapter 3: Perception & Vision ←────────┐                  │
│      ↓                                   │                  │
│  Chapter 4: Navigation & Planning        │ Parallel tracks  │
│      ↓                                   │                  │
│  Chapter 5: Manipulation & Control ←─────┘                  │
│      ↓                                                      │
│  Chapter 6: LLM Integration                                 │
│      ↓                                                      │
│  Chapter 7: Vision-Language-Action Pipelines                │
│      ↓                                                      │
│  Chapter 8: Capstone Project                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Middleware** | ROS 2 Humble | Robot communication framework |
| **Simulation** | Gazebo Sim | Physics-based robot testing |
| **Perception** | Isaac ROS | GPU-accelerated vision pipelines |
| **Navigation** | Nav2 | Autonomous navigation stack |
| **Manipulation** | MoveIt 2 | Motion planning framework |
| **LLM** | Ollama (default) | Local language model inference |
| **Languages** | Python 3.10+, C++17 | Implementation languages |

## Capstone Preview

The final project integrates all chapters into an autonomous humanoid that:

```
"Fetch the red cup from the kitchen table"
         ↓
┌─────────────────┐
│   Voice Input   │ ← Whisper speech-to-text
└────────┬────────┘
         ↓
┌─────────────────┐
│ Intent Parsing  │ ← Natural language understanding
└────────┬────────┘
         ↓
┌─────────────────┐
│ LLM Task Plan   │ ← Decompose into subtasks
└────────┬────────┘
         ↓
┌─────────────────┐
│ ROS 2 Actions   │ ← Convert to executable actions
└────────┬────────┘
         ↓
┌─────────────────┐
│   Navigation    │ ← Navigate to kitchen
└────────┬────────┘
         ↓
┌─────────────────┐
│     Vision      │ ← Detect red cup
└────────┬────────┘
         ↓
┌─────────────────┐
│  Manipulation   │ ← Grasp the cup
└────────┬────────┘
         ↓
┌─────────────────┐
│   Return        │ ← Navigate back to user
└─────────────────┘
```

## How to Use This Book

### Sequential Learning

Work through chapters in order. Each builds on previous concepts.

### Hands-On Practice

Every chapter includes:
- **Code Examples** - Runnable Python/C++ code
- **Exercises** - Practice problems with solutions
- **Checkpoints** - Verify your understanding

### Quick Start

```bash
# Clone the repository
git clone https://github.com/[org]/physical-ai-book.git
cd physical-ai-book

# Start with Docker (recommended)
docker compose up -d

# Or install locally
npm install
npm start
```

## Quality Commitment

This book follows strict quality standards:

- **Reproducibility**: Every example runs without modification
- **Completeness**: No undocumented dependencies
- **Engineering Realism**: Addresses latency, noise, and failure modes
- **Concept-First**: Focuses on ideas that survive tooling changes

## Let's Begin

Ready to build intelligent robots? Start with [Chapter 1: ROS 2 Fundamentals](/docs/chapter-01-ros2-fundamentals).
