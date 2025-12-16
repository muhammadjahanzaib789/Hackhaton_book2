# Physical AI & Humanoid Robotics

> **Embodied Intelligence in the Real World**

A comprehensive, open-source technical book teaching Physical AI and Humanoid Robotics from first principles to deployment.

## Overview

This book bridges the gap between digital AI and embodied intelligence, providing a unified learning path covering:

- **ROS 2 Fundamentals** - The robotic nervous system
- **Physics Simulation** - Gazebo Sim for realistic humanoid testing
- **Perception & Vision** - Camera, LIDAR, and object detection
- **Navigation & Planning** - SLAM, Nav2, and path planning
- **Manipulation & Control** - Inverse kinematics and MoveIt 2
- **LLM Integration** - Natural language to robot actions
- **Vision-Language-Action** - End-to-end embodied AI pipelines
- **Capstone Project** - Autonomous humanoid completing real tasks

## Core Thesis

> Intelligence reaches its full potential only when it is embodied. Physical AI bridges the digital brain and the physical body, enabling machines to perceive, reason, and act in the real world.

## Target Audience

- Advanced AI students transitioning to robotics
- Robotics engineers learning modern AI integration
- Researchers exploring embodied intelligence
- Engineers building humanoid robot systems

## Prerequisites

- Basic Python and/or C++ programming experience
- Linux system or Docker for ROS 2 development
- No prior robotics experience required

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/[org]/physical-ai-book.git
cd physical-ai-book
docker compose up -d
open http://localhost:3000
```

### Option 2: Local Development

```bash
git clone https://github.com/[org]/physical-ai-book.git
cd physical-ai-book
npm install
npm start
```

## Project Structure

```
physical-ai-book/
├── docs/                    # Book content (Markdown)
│   ├── intro.md
│   ├── chapter-01-ros2-fundamentals/
│   ├── chapter-02-simulation/
│   └── ...
├── static/
│   ├── img/                 # Diagrams and images
│   └── models/humanoid/     # URDF/SDF robot models
├── src/examples/            # Runnable code examples
│   ├── ros2/                # ROS 2 Python/C++ examples
│   ├── llm/                 # LLM integration code
│   └── capstone/            # Complete capstone project
└── tests/                   # Integration tests
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Documentation | Docusaurus 3.x |
| Robot Framework | ROS 2 Humble |
| Simulation | Gazebo Sim (Garden/Harmonic) |
| LLM Provider | Ollama (default), OpenAI/Anthropic |
| Languages | Python 3.10+, C++17, TypeScript |

## Learning Path

```
Chapter 1: ROS 2 Fundamentals
    ↓
Chapter 2: Simulation with Gazebo
    ↓
Chapter 3: Perception & Vision
    ↓
Chapter 4: Navigation & Planning
    ↓
Chapter 5: Manipulation & Control
    ↓
Chapter 6: LLM Integration
    ↓
Chapter 7: Vision-Language-Action
    ↓
Chapter 8: Capstone Project
```

## Capstone Project

Build an autonomous humanoid that:
1. Accepts voice commands ("fetch the red cup")
2. Uses LLM for task decomposition
3. Navigates to the target location
4. Identifies objects using computer vision
5. Grasps and manipulates objects
6. Returns to the user

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ROS 2 Community
- Open Source Robotics Foundation
- Gazebo Sim Development Team
- Anthropic (Claude Code authoring system)
