---
sidebar_position: 3
title: "Lesson 3: Physics Configuration"
description: "Tuning physics engines for humanoid simulation"
---

# Physics Configuration

## Learning Objectives

By the end of this lesson, you will be able to:

1. Configure physics engine parameters for stability
2. Tune contact and friction models
3. Optimize simulation performance
4. Debug physics-related issues

## Prerequisites

- Completed Lessons 1-2 of this chapter
- Humanoid URDF loaded in Gazebo
- Understanding of basic physics concepts

## Physics Engine Overview

Gazebo Sim supports multiple physics engines, each with trade-offs:

```
┌─────────────────────────────────────────────────────────────┐
│                Physics Engine Comparison                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Engine   │ Accuracy │ Speed │ Best For                    │
│  ─────────┼──────────┼───────┼────────────────────────    │
│  DART     │ ★★★★★   │ ★★★   │ Humanoids, manipulation     │
│  Bullet   │ ★★★★    │ ★★★★★ │ Mobile robots, games        │
│  ODE      │ ★★★     │ ★★★★  │ General purpose             │
│                                                             │
│  Recommendation for humanoids: DART                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Step Size and Stability

The simulation step size is the most critical parameter:

### Understanding Step Size

```
┌─────────────────────────────────────────────────────────────┐
│              Step Size and Accuracy                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Time ─────────────────────────────────────────────▶        │
│                                                             │
│  Large step (10ms):                                         │
│  │─────────│─────────│─────────│                           │
│  ▲         ▲         ▲         ▲   Few samples, fast       │
│                                    but may miss events      │
│                                                             │
│  Small step (1ms):                                          │
│  │─│─│─│─│─│─│─│─│─│─│─│─│                                │
│  ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲   Many samples, accurate      │
│                                    but computationally      │
│                                    expensive                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Choosing Step Size

| Robot Type | Recommended Step | Reason |
|------------|-----------------|--------|
| Humanoid (balance) | 0.001s (1ms) | Fast controller response |
| Arm manipulation | 0.002s (2ms) | Moderate precision |
| Mobile robot | 0.005s (5ms) | Less dynamic |
| Simple demo | 0.01s (10ms) | Fast visualization |

### SDF Configuration

```xml
<physics name="humanoid_physics" type="dart">
  <!-- Step size in seconds -->
  <max_step_size>0.001</max_step_size>

  <!-- Target real-time factor (1.0 = real-time) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Max contacts per collision pair -->
  <max_contacts>10</max_contacts>

  <!-- DART-specific configuration -->
  <dart>
    <collision_detector>bullet</collision_detector>
    <solver>
      <solver_type>dantzig</solver_type>
    </solver>
  </dart>
</physics>
```

## Contact and Friction

Contact physics determines how objects interact:

### Contact Model

```
┌─────────────────────────────────────────────────────────────┐
│                   Contact Model                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│           Normal Force (Fn)                                 │
│               ▲                                             │
│               │                                             │
│       ┌───────┴───────┐                                    │
│       │    Object     │                                    │
│       └───────────────┘                                    │
│  ◀─────────────────────▶  Friction (Ff = μ * Fn)          │
│       ═══════════════════                                  │
│           Ground                                            │
│                                                             │
│  Contact Parameters:                                        │
│  - μ (mu): Friction coefficient                            │
│  - kp: Stiffness (spring constant)                         │
│  - kd: Damping                                             │
│  - max_vel: Maximum correction velocity                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Friction Configuration

```xml
<collision name="foot_collision">
  <geometry>
    <box>
      <size>0.2 0.1 0.03</size>
    </box>
  </geometry>

  <surface>
    <friction>
      <ode>
        <!-- Friction coefficients -->
        <mu>1.0</mu>     <!-- Primary friction -->
        <mu2>1.0</mu2>   <!-- Secondary friction -->

        <!-- Friction direction (optional) -->
        <fdir1>1 0 0</fdir1>

        <!-- Slip coefficients -->
        <slip1>0.0</slip1>
        <slip2>0.0</slip2>
      </ode>

      <torsional>
        <coefficient>0.1</coefficient>
        <patch_radius>0.05</patch_radius>
      </torsional>
    </friction>

    <contact>
      <ode>
        <!-- Contact stiffness -->
        <kp>1e6</kp>

        <!-- Contact damping -->
        <kd>100</kd>

        <!-- Maximum correcting velocity -->
        <max_vel>100</max_vel>

        <!-- Minimum contact depth -->
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Friction Coefficient Guide

| Surface Pair | μ (mu) | Description |
|--------------|--------|-------------|
| Rubber on concrete | 0.8-1.0 | Robot feet on floor |
| Rubber on metal | 0.6-0.8 | Gripper on objects |
| Plastic on plastic | 0.3-0.5 | Low friction |
| Lubricated metal | 0.1-0.2 | Joints |

:::tip Humanoid Feet
For stable bipedal walking, use high friction (μ ≥ 0.8) for foot contacts. Too low friction causes slipping; too high can cause unrealistic sticking.
:::

## Joint Dynamics

Joint friction and damping affect control behavior:

### Joint Parameters

```xml
<joint name="left_hip_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <axis xyz="0 1 0"/>

  <limit lower="-1.57" upper="1.57" effort="150" velocity="6"/>

  <!-- Dynamics parameters -->
  <dynamics>
    <!-- Viscous damping coefficient -->
    <damping>0.5</damping>

    <!-- Coulomb friction -->
    <friction>0.1</friction>

    <!-- Spring reference position (for compliant joints) -->
    <spring_reference>0</spring_reference>

    <!-- Spring stiffness -->
    <spring_stiffness>0</spring_stiffness>
  </dynamics>
</joint>
```

### Damping and Friction Effects

| Parameter | Effect | Humanoid Typical |
|-----------|--------|------------------|
| `damping` | Velocity-dependent resistance | 0.1-1.0 N·m·s/rad |
| `friction` | Static/kinetic resistance | 0.0-0.2 N·m |

```
┌─────────────────────────────────────────────────────────────┐
│                Joint Damping Effect                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Velocity                                                   │
│      ▲                                                      │
│      │     No damping                                       │
│      │    ╱────────────────                                │
│      │   ╱                                                  │
│      │  ╱   With damping                                   │
│      │ ╱   ╱────────────────                               │
│      │╱  ╱                                                  │
│      └──────────────────────▶ Time                         │
│                                                             │
│  Damping smooths motion and absorbs oscillations           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Gravity and Environment

### Setting Gravity

```xml
<world name="humanoid_world">
  <!-- Standard Earth gravity -->
  <gravity>0 0 -9.81</gravity>

  <!-- Magnetic field (for compass sensors) -->
  <magnetic_field>5.5645e-6 22.8758e-6 -42.3884e-6</magnetic_field>

  <!-- Atmosphere (for drag) -->
  <atmosphere type="adiabatic"/>

  <!-- Wind (affects flying/falling) -->
  <wind>
    <linear_velocity>0 0 0</linear_velocity>
  </wind>
</world>
```

### Gravity for Different Scenarios

| Scenario | Gravity | Use Case |
|----------|---------|----------|
| Earth | (0, 0, -9.81) | Normal simulation |
| Moon | (0, 0, -1.62) | Lunar robotics |
| Mars | (0, 0, -3.72) | Mars missions |
| Zero-G | (0, 0, 0) | Space simulation |
| Reduced | (0, 0, -4.9) | Balance training |

## Performance Optimization

### Profiling Simulation

```bash
# Run Gazebo with profiling
gz sim --verbose 4 humanoid_world.sdf

# Check real-time factor
gz stats

# Output:
# Real-time factor: 0.85
# Sim time: 10.0s
# Real time: 11.8s
# Iterations: 10000
```

### Optimization Strategies

| Issue | Solution | Impact |
|-------|----------|--------|
| Low RTF | Increase step size | -Accuracy +Speed |
| Collision slow | Use primitives | +Speed |
| Many contacts | Reduce max_contacts | +Speed |
| Physics unstable | Decrease step size | +Stability |

### Multi-threaded Physics

```xml
<physics name="parallel_physics" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>

  <!-- Enable parallel computation -->
  <dart>
    <collision_detector>bullet</collision_detector>

    <!-- Use multiple threads -->
    <solver>
      <solver_type>pgs</solver_type>
      <island_threads>4</island_threads>
    </solver>
  </dart>
</physics>
```

## Debugging Physics Issues

### Common Problems and Solutions

#### 1. Robot Falls Through Ground

**Symptom**: Robot passes through floor
**Cause**: Collision not detected
**Solution**:
```xml
<!-- Ensure collision geometry is defined -->
<collision name="foot_collision">
  <geometry>
    <box>
      <size>0.2 0.1 0.03</size>
    </box>
  </geometry>
</collision>
```

#### 2. Unstable Joints

**Symptom**: Joints oscillate or explode
**Cause**: Step size too large or bad inertia
**Solution**:
```xml
<!-- Reduce step size -->
<max_step_size>0.0005</max_step_size>

<!-- Add joint damping -->
<dynamics damping="1.0" friction="0.1"/>
```

#### 3. Robot Slides on Ground

**Symptom**: Feet slip when walking
**Cause**: Friction too low
**Solution**:
```xml
<surface>
  <friction>
    <ode>
      <mu>1.5</mu>
      <mu2>1.5</mu2>
    </ode>
  </friction>
</surface>
```

### Physics Debug Visualization

```bash
# Enable physics visualization in Gazebo
gz sim --physics-debug humanoid_world.sdf
```

Shows:
- Contact points (green spheres)
- Contact normals (red arrows)
- Collision geometry (wireframe)

## Complete World Configuration

Here's a complete world file optimized for humanoid simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="humanoid_world">

    <!-- Physics: Optimized for humanoid -->
    <physics name="humanoid_physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <max_contacts>20</max_contacts>

      <dart>
        <collision_detector>bullet</collision_detector>
        <solver>
          <solver_type>dantzig</solver_type>
        </solver>
      </dart>
    </physics>

    <!-- Gravity -->
    <gravity>0 0 -9.81</gravity>

    <!-- Required plugins -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground with appropriate friction -->
    <model name="ground">
      <static>true</static>
      <link name="ground_link">
        <collision name="ground_collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
            <contact>
              <ode>
                <kp>1e6</kp>
                <kd>100</kd>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name="ground_visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Summary

Key takeaways from this lesson:

1. **Step size** is the most critical physics parameter
2. **DART engine** is recommended for humanoids
3. **Contact friction** must be tuned for stable locomotion
4. **Joint damping** smooths motion and absorbs oscillations
5. **Performance** can be improved through various optimizations

## Next Steps

In the [next lesson](./lesson-04-sensor-simulation.md), we will:
- Add simulated sensors to our humanoid
- Configure cameras, LIDAR, and IMU
- Bridge sensor data to ROS 2

## Additional Resources

- [Gazebo Physics Tutorial](https://gazebosim.org/docs/harmonic/physics)
- [DART Physics Engine](https://dartsim.github.io/)
- [Contact Dynamics Theory](https://manipulation.csail.mit.edu/force.html)
