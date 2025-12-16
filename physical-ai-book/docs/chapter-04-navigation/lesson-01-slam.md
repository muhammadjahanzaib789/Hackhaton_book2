---
sidebar_position: 1
title: "Lesson 1: SLAM Fundamentals"
description: "Simultaneous Localization and Mapping for robot navigation"
---

# SLAM Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the SLAM problem and its importance
2. Differentiate between SLAM algorithms
3. Run SLAM with LIDAR data in ROS 2
4. Build and save maps for navigation

## Prerequisites

- Completed Chapter 3 (Perception)
- Understanding of robot coordinate frames
- LIDAR sensor publishing data

## What is SLAM?

SLAM (Simultaneous Localization and Mapping) solves a chicken-and-egg problem:

- **Localization**: Where am I? (requires a map)
- **Mapping**: What does the environment look like? (requires knowing where I am)

```
┌─────────────────────────────────────────────────────────────┐
│                    The SLAM Problem                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Without SLAM:                                             │
│   ┌────────────┐      ┌────────────┐                       │
│   │ Need map   │◀────▶│ Need pose  │  ← Circular!          │
│   │ to localize│      │ to map     │                       │
│   └────────────┘      └────────────┘                       │
│                                                             │
│   With SLAM:                                                │
│   ┌────────────────────────────────────────────────────┐   │
│   │            Estimate Both Simultaneously            │   │
│   │                                                    │   │
│   │   Sensor ──▶ [ SLAM Algorithm ] ──▶ Map + Pose    │   │
│   │   Data                                            │   │
│   └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why SLAM Matters for Humanoids

| Capability | Requires SLAM | Example |
|------------|---------------|---------|
| Autonomous navigation | Yes | Go to the kitchen |
| Object search | Yes | Find the red cup |
| Return to base | Yes | Come back to charging station |
| Avoid obstacles | Partial | Dynamic obstacle avoidance |

## Types of SLAM

### 1. 2D LIDAR SLAM

Uses 2D laser scans to build occupancy grid maps.

```
┌─────────────────────────────────────────────────────────────┐
│                    2D LIDAR SLAM                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   LIDAR Scan              Occupancy Grid                    │
│                                                             │
│        ╱╲                 ░░░░░░░░░░░░░░                   │
│       ╱  ╲                ░░██████░░░░░░                   │
│      ╱    ╲               ░░░░░░░░░░░░░░                   │
│     ╱      ╲              ░░░░░░░░██████                   │
│    ╱   R    ╲             ░░░░░R░░░░░░░░                   │
│     ╲      ╱              ░░░░░░░░░░░░░░                   │
│      ╲    ╱               ░░██████░░░░░░                   │
│       ╲  ╱                ░░░░░░░░░░░░░░                   │
│        ╲╱                                                  │
│                           R = Robot                        │
│   Range measurements      █ = Obstacle                     │
│   in a plane              ░ = Free space                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Popular Algorithms:**
- **slam_toolbox** (ROS 2 default)
- **Cartographer** (Google)
- **Hector SLAM**

### 2. Visual SLAM (VSLAM)

Uses camera images for mapping.

**Popular Algorithms:**
- **ORB-SLAM3**
- **RTAB-Map**
- **Isaac ROS Visual SLAM**

### 3. 3D LIDAR SLAM

Uses 3D point clouds for full environment modeling.

**Popular Algorithms:**
- **LIO-SAM**
- **LOAM**
- **hdl_graph_slam**

## SLAM with slam_toolbox

slam_toolbox is the recommended SLAM package for ROS 2.

### Installation

```bash
# Install slam_toolbox
sudo apt install ros-humble-slam-toolbox

# Verify
ros2 pkg list | grep slam
```

### Configuration

Create `slam_params.yaml`:

```yaml
slam_toolbox:
  ros__parameters:
    # Plugin params
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT
    ceres_dogleg_type: TRADITIONAL_DOGLEG
    ceres_loss_function: None

    # ROS Parameters
    odom_frame: odom
    map_frame: map
    base_frame: base_link
    scan_topic: /scan
    mode: mapping  # or localization

    # Processing
    debug_logging: false
    throttle_scans: 1
    transform_publish_period: 0.02
    map_update_interval: 5.0
    resolution: 0.05
    max_laser_range: 20.0
    minimum_time_interval: 0.5
    transform_timeout: 0.2
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000

    # General Parameters
    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.5
    minimum_travel_heading: 0.5
    scan_buffer_size: 10
    scan_buffer_maximum_scan_distance: 10.0
    link_match_minimum_response_fine: 0.1
    link_scan_maximum_distance: 1.5
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_coarse: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    # Correlation Parameters
    correlation_search_space_dimension: 0.5
    correlation_search_space_resolution: 0.01
    correlation_search_space_smear_deviation: 0.1

    # Loop Closure Parameters
    loop_search_space_dimension: 8.0
    loop_search_space_resolution: 0.05
    loop_search_space_smear_deviation: 0.03

    # Scan Matcher Parameters
    distance_variance_penalty: 0.5
    angle_variance_penalty: 1.0
    fine_search_angle_offset: 0.00349
    coarse_search_angle_offset: 0.349
    coarse_angle_resolution: 0.0349
    minimum_angle_penalty: 0.9
    minimum_distance_penalty: 0.5
    use_response_expansion: true
```

### Launch SLAM

```python
#!/usr/bin/env python3
"""
SLAM Launch File
Physical AI Book - Chapter 4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    slam_params_file = os.path.join(
        get_package_share_directory('physical_ai_book'),
        'config', 'slam_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),

        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_params_file,
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
```

### Running SLAM

```bash
# Terminal 1: Launch simulation
ros2 launch physical_ai_book humanoid_sim.launch.py

# Terminal 2: Launch SLAM
ros2 launch physical_ai_book slam.launch.py

# Terminal 3: Teleoperate the robot to explore
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 4: Visualize in RViz
ros2 run rviz2 rviz2 -d slam.rviz
```

## Understanding the Map

### Occupancy Grid

The map is represented as an occupancy grid:

```
┌─────────────────────────────────────────────────────────────┐
│                  Occupancy Grid Values                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Value     Meaning          Visual                         │
│  ─────────────────────────────────────────                 │
│  -1        Unknown          Gray (░)                       │
│  0         Free space       White ( )                      │
│  100       Occupied         Black (█)                      │
│  1-99      Probability      Gradient                       │
│                                                             │
│  Example Grid (10x10):                                     │
│                                                             │
│    ░░░░░░░░░░                                              │
│    ░░████░░░░                                              │
│    ░░    ░░░░    █ = Wall (value: 100)                    │
│    ░░    ░░░░      = Free (value: 0)                      │
│    ░░░░░░░░░░    ░ = Unknown (value: -1)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Map Metadata

```yaml
# map.yaml (saved with map)
image: map.pgm
resolution: 0.05  # meters per pixel
origin: [-10.0, -10.0, 0.0]  # [x, y, yaw]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

## Saving and Loading Maps

### Saving a Map

```bash
# Save current map
ros2 run nav2_map_server map_saver_cli -f my_map

# This creates:
# - my_map.pgm (image)
# - my_map.yaml (metadata)
```

### Loading a Map

```python
# In launch file
Node(
    package='nav2_map_server',
    executable='map_server',
    name='map_server',
    parameters=[{
        'yaml_filename': '/path/to/my_map.yaml',
        'use_sim_time': True,
    }],
),
```

## Localization Mode

Once you have a map, switch to localization-only mode:

```yaml
# In slam_params.yaml
slam_toolbox:
  ros__parameters:
    mode: localization  # Changed from 'mapping'
```

Or use AMCL for pure localization:

```bash
# Launch AMCL
ros2 launch nav2_bringup localization_launch.py \
    map:=/path/to/my_map.yaml \
    use_sim_time:=true
```

## SLAM Quality Metrics

### Good SLAM Characteristics

| Metric | Good | Bad |
|--------|------|-----|
| Loop closures | Many, consistent | Few or inconsistent |
| Map clarity | Sharp walls | Fuzzy/double walls |
| Drift | Minimal | Accumulating |
| Processing time | Real-time | Lagging |

### Debugging SLAM Issues

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Expected transforms:
# map -> odom -> base_link -> sensors

# Monitor SLAM status
ros2 topic echo /slam_toolbox/feedback
```

## Summary

Key takeaways from this lesson:

1. **SLAM** solves localization and mapping simultaneously
2. **slam_toolbox** is the standard for ROS 2
3. **Occupancy grids** represent environment as probabilities
4. **Maps can be saved** for later navigation
5. **Quality depends** on sensor data and motion

## Next Steps

In the [next lesson](./lesson-02-nav2.md), we will:
- Set up the Nav2 navigation stack
- Configure path planners
- Implement autonomous navigation

## Additional Resources

- [slam_toolbox Documentation](https://github.com/SteveMacenski/slam_toolbox)
- [ROS 2 Navigation](https://navigation.ros.org/)
- [Cartographer](https://google-cartographer-ros.readthedocs.io/)
