---
sidebar_position: 3
title: "Lesson 3: Path Planning Algorithms"
description: "Understanding and implementing path planning for robots"
---

# Path Planning Algorithms

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand fundamental path planning algorithms
2. Compare A*, Dijkstra, and RRT planners
3. Configure Nav2 planners for different scenarios
4. Implement custom planning behaviors

## Prerequisites

- Completed Lessons 1-2 of this chapter
- Understanding of graphs and search algorithms
- Nav2 running with a map

## Path Planning Overview

Path planning finds a collision-free path from start to goal.

```
┌─────────────────────────────────────────────────────────────┐
│                Path Planning Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Start ──▶ Search Space ──▶ Algorithm ──▶ Path ──▶ Goal   │
│  Pose      (Costmap)        (A*, RRT)     (waypoints)      │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Map with Obstacles                                 │  │
│   │                                                     │  │
│   │   S ─ ─ ─ ─ ┐    ████████                         │  │
│   │             │    ████████                         │  │
│   │             │         │                           │  │
│   │   ██████    └ ─ ─ ─ ─ ┼ ─ ─ ┐                    │  │
│   │   ██████              │     │                     │  │
│   │                       │     │                     │  │
│   │   ████████████████    │     └ ─ ─ G              │  │
│   │                       │                           │  │
│   │   S = Start          G = Goal                    │  │
│   │   ─ = Path           █ = Obstacle                │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Fundamental Algorithms

### Dijkstra's Algorithm

Finds the shortest path by expanding in all directions.

```python
def dijkstra(graph, start, goal):
    """
    Dijkstra's shortest path algorithm.

    Expands nodes in order of distance from start.
    Guarantees optimal path but explores many nodes.
    """
    import heapq

    # Priority queue: (distance, node)
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current_dist, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor, edge_cost in graph.neighbors(current):
            tentative_g = g_score[current] + edge_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))

    return None  # No path found
```

### A* Algorithm

Improves on Dijkstra by using a heuristic to guide the search.

```python
def astar(graph, start, goal, heuristic):
    """
    A* search algorithm.

    Uses heuristic (estimated distance to goal) to prioritize
    nodes that seem closer to the goal.

    Optimal if heuristic is admissible (never overestimates).
    """
    import heapq

    # Priority queue: (f_score, node)
    # f_score = g_score + heuristic
    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor, edge_cost in graph.neighbors(current):
            tentative_g = g_score[current] + edge_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


def euclidean_heuristic(a, b):
    """Euclidean distance heuristic."""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def manhattan_heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

### Algorithm Comparison

| Algorithm | Optimal | Complete | Time Complexity | Use Case |
|-----------|---------|----------|-----------------|----------|
| **Dijkstra** | Yes | Yes | O(V²) | Uniform cost |
| **A*** | Yes* | Yes | O(E) | Most scenarios |
| **RRT** | No | Prob. | Varies | High dimensions |
| **RRT*** | Yes | Prob. | Higher | Quality paths |

*With admissible heuristic

## RRT (Rapidly-exploring Random Trees)

RRT is effective for high-dimensional spaces and complex constraints.

```python
import random
import math


class RRT:
    """
    Rapidly-exploring Random Tree planner.

    Good for:
    - High-dimensional configuration spaces
    - Complex kinematic constraints
    - Non-convex obstacles
    """

    def __init__(self, start, goal, obstacle_checker, bounds, step_size=0.5):
        self.start = start
        self.goal = goal
        self.is_collision_free = obstacle_checker
        self.bounds = bounds  # [(x_min, x_max), (y_min, y_max)]
        self.step_size = step_size

        # Tree: node -> parent
        self.tree = {start: None}

    def plan(self, max_iterations=1000, goal_bias=0.1):
        """
        Plan a path using RRT.

        Args:
            max_iterations: Maximum planning iterations
            goal_bias: Probability of sampling the goal

        Returns:
            Path as list of points, or None
        """
        for _ in range(max_iterations):
            # Sample random point (with goal bias)
            if random.random() < goal_bias:
                sample = self.goal
            else:
                sample = self.random_sample()

            # Find nearest node in tree
            nearest = self.nearest_node(sample)

            # Extend toward sample
            new_node = self.extend(nearest, sample)

            if new_node and self.is_collision_free(nearest, new_node):
                self.tree[new_node] = nearest

                # Check if we reached the goal
                if self.distance(new_node, self.goal) < self.step_size:
                    if self.is_collision_free(new_node, self.goal):
                        self.tree[self.goal] = new_node
                        return self.extract_path()

        return None  # No path found

    def random_sample(self):
        """Generate random sample within bounds."""
        x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = random.uniform(self.bounds[1][0], self.bounds[1][1])
        return (x, y)

    def nearest_node(self, point):
        """Find nearest node in tree to point."""
        return min(self.tree.keys(), key=lambda n: self.distance(n, point))

    def extend(self, from_node, to_point):
        """Extend from node toward point by step_size."""
        dist = self.distance(from_node, to_point)
        if dist < self.step_size:
            return to_point

        ratio = self.step_size / dist
        x = from_node[0] + ratio * (to_point[0] - from_node[0])
        y = from_node[1] + ratio * (to_point[1] - from_node[1])
        return (x, y)

    def distance(self, a, b):
        """Euclidean distance between two points."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def extract_path(self):
        """Extract path from tree."""
        path = [self.goal]
        current = self.goal

        while self.tree[current] is not None:
            current = self.tree[current]
            path.append(current)

        return list(reversed(path))
```

## Nav2 Planners

### NavFn Planner (Default)

Grid-based planner using Dijkstra or A*.

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: true  # Use A* instead of Dijkstra
      allow_unknown: true
```

### Smac Planner (Hybrid A*)

Better for non-holonomic robots (cars, differential drive).

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["SmacPlanner"]
    SmacPlanner:
      plugin: "nav2_smac_planner/SmacPlannerHybrid"
      tolerance: 0.25
      downsample_costmap: false
      downsampling_factor: 1
      allow_unknown: true
      max_iterations: 1000000
      max_on_approach_iterations: 1000
      max_planning_time: 5.0
      motion_model_for_search: "DUBIN"  # or "REEDS_SHEPP"
      cost_travel_multiplier: 2.0
      angle_quantization_bins: 72
      analytic_expansion_ratio: 3.5
      analytic_expansion_max_length: 3.0
      minimum_turning_radius: 0.4
      reverse_penalty: 2.0
      change_penalty: 0.0
      non_straight_penalty: 1.2
      cost_penalty: 2.0
      retrospective_penalty: 0.015
      lookup_table_size: 20.0
```

### Theta* Planner

Produces smoother, any-angle paths.

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["ThetaStar"]
    ThetaStar:
      plugin: "nav2_theta_star_planner/ThetaStarPlanner"
      how_many_corners: 8
      w_euc_cost: 1.0
      w_traversal_cost: 2.0
      w_heuristic_cost: 1.0
```

## Local Planners (Controllers)

Local planners handle trajectory following and obstacle avoidance.

### DWB (Dynamic Window Approach)

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      min_vel_x: 0.0
      max_vel_x: 0.5
      max_vel_theta: 1.0
      acc_lim_x: 2.5
      acc_lim_theta: 3.2
      sim_time: 1.7
      critics: ["RotateToGoal", "Oscillation", "ObstacleFootprint", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
```

### MPPI (Model Predictive Path Integral)

Advanced controller using optimization.

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 56
      model_dt: 0.05
      batch_size: 2000
      vx_std: 0.2
      vy_std: 0.2
      wz_std: 0.4
      vx_max: 0.5
      vx_min: -0.35
      vy_max: 0.5
      wz_max: 1.9
```

## Custom Planning

### Simple Path Smoothing

```python
def smooth_path(path, iterations=5, weight_data=0.5, weight_smooth=0.3):
    """
    Smooth a path using gradient descent.

    Args:
        path: List of (x, y) waypoints
        iterations: Smoothing iterations
        weight_data: Pull toward original path
        weight_smooth: Pull toward neighbors

    Returns:
        Smoothed path
    """
    import numpy as np

    # Convert to numpy array
    path = np.array(path, dtype=float)
    smoothed = path.copy()

    for _ in range(iterations):
        for i in range(1, len(path) - 1):
            # Data term: stay close to original
            data_pull = weight_data * (path[i] - smoothed[i])

            # Smooth term: move toward midpoint of neighbors
            smooth_pull = weight_smooth * (
                smoothed[i-1] + smoothed[i+1] - 2 * smoothed[i]
            )

            smoothed[i] += data_pull + smooth_pull

    return smoothed.tolist()
```

### Velocity Profiling

```python
def compute_velocity_profile(path, max_vel, max_accel, max_decel):
    """
    Compute velocity profile for smooth motion.

    Uses trapezoidal velocity profile with acceleration limits.
    """
    import numpy as np

    # Compute segment lengths
    segments = []
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        segments.append(np.sqrt(dx**2 + dy**2))

    total_length = sum(segments)

    # Trapezoidal profile
    # Accelerate, cruise, decelerate
    accel_dist = max_vel**2 / (2 * max_accel)
    decel_dist = max_vel**2 / (2 * max_decel)

    if accel_dist + decel_dist > total_length:
        # Triangle profile (never reach max vel)
        peak_vel = np.sqrt(2 * max_accel * max_decel * total_length /
                          (max_accel + max_decel))
        accel_dist = peak_vel**2 / (2 * max_accel)
        cruise_dist = 0
    else:
        peak_vel = max_vel
        cruise_dist = total_length - accel_dist - decel_dist

    # Assign velocities to waypoints
    velocities = []
    traveled = 0

    for i, seg_len in enumerate(segments):
        if traveled < accel_dist:
            # Accelerating
            vel = np.sqrt(2 * max_accel * traveled)
        elif traveled < accel_dist + cruise_dist:
            # Cruising
            vel = peak_vel
        else:
            # Decelerating
            remaining = total_length - traveled
            vel = np.sqrt(2 * max_decel * remaining)

        velocities.append(min(vel, max_vel))
        traveled += seg_len

    velocities.append(0)  # Stop at goal
    return velocities
```

## Summary

Key takeaways from this lesson:

1. **A*** is the go-to algorithm for grid-based planning
2. **RRT** handles high-dimensional and constrained spaces
3. **Nav2 planners** offer different trade-offs
4. **Local planners** handle real-time obstacle avoidance
5. **Post-processing** improves path quality

## Next Steps

Continue to [Chapter 5: Manipulation](../chapter-05-manipulation/lesson-01-kinematics.md) to learn:
- Robot arm kinematics
- Motion planning with MoveIt2
- Grasping and manipulation

## Additional Resources

- [Planning Algorithms Book](http://planning.cs.uiuc.edu/)
- [Nav2 Planner Plugins](https://navigation.ros.org/plugins/index.html)
- [OMPL Library](https://ompl.kavrakilab.org/)
