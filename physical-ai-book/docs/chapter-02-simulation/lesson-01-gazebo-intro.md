---
sidebar_position: 1
title: "Lesson 1: Introduction to Gazebo Sim"
description: "Understanding Gazebo Sim for robotics simulation"
---

# Introduction to Gazebo Sim

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand Gazebo Sim's role in robotics development
2. Navigate the Gazebo Sim interface effectively
3. Load and interact with robot models
4. Configure basic simulation parameters

## Prerequisites

- Completed Chapter 1 (ROS 2 Fundamentals)
- Docker environment or native Ubuntu 22.04 installation
- Basic understanding of 3D coordinate systems

## What is Gazebo Sim?

Gazebo Sim (formerly Ignition Gazebo) is a next-generation robotics simulator that provides:

- **Realistic Physics**: Multiple physics engines (DART, Bullet, ODE)
- **High-Fidelity Sensors**: Camera, LIDAR, IMU, depth sensors
- **ROS 2 Integration**: Native bridge to ROS 2 topics and services
- **Scalability**: Distributed simulation support
- **Modern Architecture**: Plugin-based, modular design

### Why Simulation Matters for Physical AI

```
┌─────────────────────────────────────────────────────────────┐
│                 Simulation-to-Real Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Develop  │───▶│ Simulate │───▶│ Deploy   │            │
│   │ in Sim   │    │ & Test   │    │ to Real  │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│        │               │               │                   │
│        ▼               ▼               ▼                   │
│   - Safe testing  - Validate    - Confident               │
│   - Fast iteration  behavior      deployment              │
│   - No hardware   - Find bugs   - Known limits            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

For humanoid robotics, simulation is essential because:

1. **Safety**: Testing balance and locomotion without risking expensive hardware
2. **Speed**: Running thousands of experiments in parallel
3. **Reproducibility**: Deterministic environments for debugging
4. **Sensor Synthesis**: Generating training data for perception models

## Gazebo Sim Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gazebo Sim Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Server (gzserver)                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Physics │  │ Sensors │  │ Plugins │            │   │
│  │  │ Engine  │  │ System  │  │ System  │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          │ Transport Layer                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Client (gzclient)                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ 3D View │  │ Tools   │  │ Widgets │            │   │
│  │  │ Render  │  │ Panel   │  │ Panel   │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **World** | Environment definition | Ground plane, lighting |
| **Models** | Robot and object definitions | Humanoid URDF/SDF |
| **Plugins** | Extend functionality | Sensors, controllers |
| **Physics** | Dynamics simulation | ODE, DART, Bullet |
| **Rendering** | Visualization | OGRE2 engine |

## Starting Gazebo Sim

### Using Docker (Recommended)

```bash
# Start the development container
docker-compose up -d ros2

# Enter the container
docker-compose exec ros2 bash

# Launch Gazebo with empty world
gz sim empty.sdf
```

### Native Installation

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Launch Gazebo
gz sim empty.sdf
```

### Expected Output

When Gazebo starts, you'll see:

```
[INFO] [gz-sim-server]: Loading SDF world file: empty.sdf
[INFO] [gz-sim-server]: Starting physics engine: dart
[INFO] [gz-sim-server]: Physics update rate: 1000 Hz
[GUI] [gz-sim-gui]: Loading plugins...
```

## Gazebo Sim Interface

### Main Window

The Gazebo Sim interface consists of:

1. **3D Viewport**: Central area showing the simulation world
2. **Entity Tree**: Left panel listing all entities
3. **Component Inspector**: Right panel showing entity properties
4. **Toolbar**: Top bar with tools and controls
5. **Plugin Panel**: Bottom area for additional plugins

### Navigation Controls

| Action | Mouse | Keyboard |
|--------|-------|----------|
| **Orbit** | Left-drag | - |
| **Pan** | Middle-drag | Shift + Left-drag |
| **Zoom** | Scroll wheel | - |
| **Reset View** | - | R |
| **Top View** | - | 5 |
| **Front View** | - | 1 |

### Simulation Controls

| Control | Shortcut | Description |
|---------|----------|-------------|
| **Play** | Space | Start simulation |
| **Pause** | Space | Pause simulation |
| **Step** | Right Arrow | Single step forward |
| **Reset** | Ctrl+R | Reset to initial state |

## Loading Our Humanoid Model

Let's load the humanoid robot we defined in Chapter 1.

### Create a World File

Create a new file `humanoid_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="humanoid_world">

    <!-- Physics configuration -->
    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Plugins -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics">
    </plugin>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster">
    </plugin>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include humanoid model -->
    <include>
      <uri>model://humanoid</uri>
      <pose>0 0 0.95 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### Launch the World

```bash
# Set model path
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/physical-ai-book/static/models

# Launch
gz sim humanoid_world.sdf
```

## Simulation Time vs. Wall Time

Understanding time in simulation is crucial:

```
┌─────────────────────────────────────────────────────────────┐
│                    Time Concepts                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Wall Time: Real elapsed time                               │
│  └── 10 seconds = 10 seconds on your clock                 │
│                                                             │
│  Sim Time: Simulated elapsed time                           │
│  └── Can run faster/slower than real time                  │
│                                                             │
│  Real-Time Factor (RTF):                                    │
│  └── RTF = Sim Time / Wall Time                            │
│  └── RTF = 1.0: Real-time                                  │
│  └── RTF = 2.0: 2x faster than real-time                   │
│  └── RTF = 0.5: Half real-time (complex sim)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Checking Performance

```bash
# View simulation stats
gz stats

# Expected output:
# Real-time factor: 0.98
# Sim time: 10.5s
# Wall time: 10.7s
```

:::tip Real-Time Factor
For humanoid simulation with full physics, aim for RTF > 0.5. Lower values indicate the simulation is struggling to keep up. Reduce sensor rates or simplify collision meshes to improve performance.
:::

## Physics Engine Selection

Gazebo Sim supports multiple physics engines:

| Engine | Strengths | Best For |
|--------|-----------|----------|
| **DART** | Accurate joint constraints | Humanoids, manipulation |
| **Bullet** | Fast, game-oriented | Mobile robots, many objects |
| **ODE** | Well-tested, stable | General purpose |

### Configuring Physics

```xml
<physics name="physics_config" type="dart">
  <!-- Step size: smaller = more accurate but slower -->
  <max_step_size>0.001</max_step_size>

  <!-- Target real-time factor -->
  <real_time_factor>1.0</real_time_factor>

  <!-- DART-specific settings -->
  <dart>
    <collision_detector>bullet</collision_detector>
    <solver>
      <solver_type>pgs</solver_type>
    </solver>
  </dart>
</physics>
```

:::warning Physics Step Size
For humanoid balance control, use a step size of 0.001s (1ms) or smaller. Larger step sizes can cause instability in joint controllers and unrealistic contact behavior.
:::

## Summary

Key takeaways from this lesson:

1. **Gazebo Sim** is essential for safe, fast robotics development
2. **Simulation-to-Real** pipelines reduce deployment risk
3. **World files** (SDF) define complete simulation environments
4. **Physics engines** should be chosen based on your robot type
5. **Real-time factor** indicates simulation performance

## Next Steps

In the [next lesson](./lesson-02-urdf-deep-dive.md), we will:
- Deep dive into URDF for humanoid robots
- Configure joints with proper limits
- Add inertial properties for accurate physics

## Additional Resources

- [Gazebo Sim Documentation](https://gazebosim.org/docs)
- [SDF Format Specification](http://sdformat.org/spec)
- [ROS 2 Gazebo Integration](https://gazebosim.org/docs/harmonic/ros2_integration)
