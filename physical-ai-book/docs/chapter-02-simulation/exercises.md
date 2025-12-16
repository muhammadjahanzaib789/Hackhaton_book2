---
sidebar_position: 6
title: "Chapter 2 Exercises"
description: "Hands-on exercises for humanoid simulation"
---

# Chapter 2 Exercises

## Overview

These exercises reinforce the simulation concepts learned in Chapter 2. Complete them in order, as later exercises build on earlier ones.

**Estimated Time**: 3-4 hours
**Difficulty**: Beginner → Intermediate → Advanced

---

## Exercise 1: Custom World Creation

### Objective

Create a custom Gazebo world with obstacles for humanoid navigation testing.

### Difficulty

:star: Beginner (30 minutes)

### Instructions

1. Create a new world file `test_world.sdf`
2. Add a ground plane with checkerboard texture
3. Add 3-5 box obstacles of varying sizes
4. Add a goal marker (colored cylinder)
5. Configure appropriate lighting

### Starter Code

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="test_world">

    <!-- TODO: Add physics configuration -->

    <!-- TODO: Add plugins -->

    <!-- TODO: Add lighting -->

    <!-- Ground plane with checkerboard -->
    <model name="ground">
      <static>true</static>
      <link name="ground_link">
        <!-- TODO: Add visual with checkerboard material -->
        <!-- TODO: Add collision -->
      </link>
    </model>

    <!-- TODO: Add obstacles -->

    <!-- TODO: Add goal marker -->

  </world>
</sdf>
```

### Success Criteria

- [ ] World loads without errors
- [ ] Ground has visible checkerboard pattern
- [ ] At least 3 obstacles present
- [ ] Goal marker is clearly visible
- [ ] Physics simulation runs at RTF > 0.9

### Hints

<details>
<summary>Hint 1: Checkerboard Material</summary>

Use Gazebo's built-in checkerboard:
```xml
<material>
  <script>
    <uri>file://media/materials/scripts/gazebo.material</uri>
    <name>Gazebo/Checkerboard</name>
  </script>
</material>
```

</details>

<details>
<summary>Hint 2: Box Obstacle</summary>

```xml
<model name="obstacle_1">
  <static>true</static>
  <pose>2 1 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box><size>0.5 0.5 1.0</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>0.5 0.5 1.0</size></box>
      </geometry>
    </visual>
  </link>
</model>
```

</details>

### Solution

<details>
<summary>Click to reveal solution</summary>

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="test_world">

    <physics name="1ms" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>

    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground">
      <static>true</static>
      <link name="ground_link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal><size>20 20</size></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>20 20</size></plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_1">
      <static>true</static>
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.5 0.5 1.0</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.5 0.5 1.0</size></box></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <static>true</static>
      <pose>3 2 0.3 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>1.0 0.3 0.6</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>1.0 0.3 0.6</size></box></geometry>
          <material><ambient>0.2 0.8 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="goal">
      <static>true</static>
      <pose>5 0 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.3</radius><length>0.2</length></cylinder></geometry>
          <material><ambient>0.2 0.2 0.8 1</ambient></material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

</details>

---

## Exercise 2: Sensor Noise Calibration

### Objective

Implement a sensor noise analysis node that characterizes IMU noise.

### Difficulty

:star::star: Intermediate (45 minutes)

### Background

Real IMU sensors have noise that varies with conditions. This exercise analyzes simulated IMU noise to understand its characteristics.

### Instructions

1. Create a ROS 2 node that subscribes to IMU data
2. Collect 1000 samples while the robot is stationary
3. Calculate mean and standard deviation for each axis
4. Compare with configured noise values
5. Plot the noise distribution (optional)

### Starter Code

```python
#!/usr/bin/env python3
"""
IMU Noise Analyzer
Physical AI Book - Chapter 2 Exercise 2

Analyzes IMU noise characteristics.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np


class IMUNoiseAnalyzer(Node):
    """Analyzes IMU sensor noise."""

    def __init__(self):
        super().__init__('imu_noise_analyzer')

        self.samples = {
            'angular_x': [], 'angular_y': [], 'angular_z': [],
            'linear_x': [], 'linear_y': [], 'linear_z': [],
        }
        self.target_samples = 1000

        # TODO: Create subscription to /imu/data

        self.get_logger().info('IMU noise analyzer started')

    def imu_callback(self, msg):
        """Collect IMU samples."""
        # TODO: Store angular velocity and linear acceleration
        # TODO: Check if target samples reached
        # TODO: Analyze noise when done
        pass

    def analyze_noise(self):
        """Calculate noise statistics."""
        # TODO: Calculate mean and std for each axis
        # TODO: Log results
        # TODO: Shutdown node
        pass


def main(args=None):
    rclpy.init(args=args)
    node = IMUNoiseAnalyzer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Success Criteria

- [ ] Node collects exactly 1000 samples
- [ ] Calculates mean and std for all 6 axes
- [ ] Results logged clearly
- [ ] Mean values close to 0 (or bias if configured)
- [ ] Std values match configured noise

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
#!/usr/bin/env python3
"""
IMU Noise Analyzer - Solution
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np


class IMUNoiseAnalyzer(Node):
    def __init__(self):
        super().__init__('imu_noise_analyzer')

        self.samples = {
            'angular_x': [], 'angular_y': [], 'angular_z': [],
            'linear_x': [], 'linear_y': [], 'linear_z': [],
        }
        self.target_samples = 1000
        self.analyzed = False

        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.get_logger().info(f'Collecting {self.target_samples} IMU samples...')

    def imu_callback(self, msg):
        if self.analyzed:
            return

        self.samples['angular_x'].append(msg.angular_velocity.x)
        self.samples['angular_y'].append(msg.angular_velocity.y)
        self.samples['angular_z'].append(msg.angular_velocity.z)
        self.samples['linear_x'].append(msg.linear_acceleration.x)
        self.samples['linear_y'].append(msg.linear_acceleration.y)
        self.samples['linear_z'].append(msg.linear_acceleration.z)

        if len(self.samples['angular_x']) >= self.target_samples:
            self.analyze_noise()

    def analyze_noise(self):
        self.analyzed = True
        self.get_logger().info('=== IMU Noise Analysis ===')

        for axis, values in self.samples.items():
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            self.get_logger().info(f'{axis}: mean={mean:.6f}, std={std:.6f}')

        self.get_logger().info('=== Analysis Complete ===')
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = IMUNoiseAnalyzer()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

</details>

---

## Exercise 3: Simple Locomotion Controller

### Objective

Implement a basic walking motion by cycling leg joint positions.

### Difficulty

:star::star::star: Advanced (60+ minutes)

### Background

Bipedal walking involves coordinated motion of multiple joints. This exercise implements a simple open-loop walking pattern.

### Instructions

1. Create a joint trajectory for one walking step
2. Implement left-right leg coordination
3. Use sinusoidal patterns for smooth motion
4. Tune parameters for stable walking
5. Test in simulation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Walking Controller                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Gait Generator                                            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Phase: 0 ────────────▶ 2π ────────────▶ 0          │  │
│   │         │               │               │           │  │
│   │    Left stance     Right stance    Left stance      │  │
│   │    Right swing     Left swing      Right swing      │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│   Joint Trajectories                                        │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  hip_pitch[t] = A_hip × sin(phase + offset)         │  │
│   │  knee_pitch[t] = A_knee × sin(phase + offset)       │  │
│   │  ankle_pitch[t] = A_ankle × sin(phase + offset)     │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│   Joint Commands → Gazebo                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Starter Code

```python
#!/usr/bin/env python3
"""
Simple Walking Controller
Physical AI Book - Chapter 2 Exercise 3

Implements open-loop walking motion.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math


class WalkingController(Node):
    """Simple open-loop walking controller."""

    def __init__(self):
        super().__init__('walking_controller')

        # Gait parameters
        self.gait_frequency = 0.5  # Hz (steps per second)
        self.hip_amplitude = 0.3   # radians
        self.knee_amplitude = 0.5  # radians
        self.ankle_amplitude = 0.2 # radians

        # Phase offset between legs
        self.phase_offset = math.pi  # 180 degrees

        # Current phase
        self.phase = 0.0

        # TODO: Create publishers for each leg joint
        # Left leg: left_hip_pitch, left_knee_pitch, left_ankle_pitch
        # Right leg: right_hip_pitch, right_knee_pitch, right_ankle_pitch

        # Control loop timer (100 Hz)
        self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Walking controller started')

    def control_loop(self):
        """Generate and publish walking motion."""
        # Update phase
        dt = 0.01  # Timer period
        self.phase += 2 * math.pi * self.gait_frequency * dt
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi

        # TODO: Calculate joint positions for left leg
        # TODO: Calculate joint positions for right leg (with phase offset)
        # TODO: Publish joint commands
        pass

    def calculate_leg_joints(self, phase):
        """
        Calculate joint positions for one leg.

        Args:
            phase: Current gait phase (0 to 2π)

        Returns:
            tuple: (hip, knee, ankle) positions in radians
        """
        # TODO: Implement sinusoidal joint trajectories
        hip = 0.0
        knee = 0.0
        ankle = 0.0
        return hip, knee, ankle


def main(args=None):
    rclpy.init(args=args)
    controller = WalkingController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Success Criteria

- [ ] Robot moves forward in simulation
- [ ] Legs alternate in coordinated pattern
- [ ] Motion is smooth (no jerky movements)
- [ ] Robot maintains balance for 10+ steps
- [ ] Gait parameters are tunable

### Hints

<details>
<summary>Hint 1: Joint Trajectory</summary>

Use sinusoidal patterns with phase offsets:
```python
hip = self.hip_amplitude * math.sin(phase)
knee = self.knee_amplitude * (1 - math.cos(phase)) / 2  # Always bent
ankle = self.ankle_amplitude * math.sin(phase - math.pi/4)
```

</details>

<details>
<summary>Hint 2: Stability</summary>

For better stability:
- Keep center of mass over support foot
- Use smaller amplitudes initially
- Add slight ankle compensation

</details>

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
#!/usr/bin/env python3
"""
Walking Controller - Solution
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math


class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        self.gait_frequency = 0.4
        self.hip_amplitude = 0.25
        self.knee_amplitude = 0.4
        self.ankle_amplitude = 0.15
        self.phase_offset = math.pi
        self.phase = 0.0

        # Publishers
        self.pubs = {
            'left_hip': self.create_publisher(Float64, '/humanoid/left_hip_pitch/cmd_pos', 10),
            'left_knee': self.create_publisher(Float64, '/humanoid/left_knee_pitch/cmd_pos', 10),
            'left_ankle': self.create_publisher(Float64, '/humanoid/left_ankle_pitch/cmd_pos', 10),
            'right_hip': self.create_publisher(Float64, '/humanoid/right_hip_pitch/cmd_pos', 10),
            'right_knee': self.create_publisher(Float64, '/humanoid/right_knee_pitch/cmd_pos', 10),
            'right_ankle': self.create_publisher(Float64, '/humanoid/right_ankle_pitch/cmd_pos', 10),
        }

        self.create_timer(0.01, self.control_loop)
        self.get_logger().info('Walking controller ready')

    def control_loop(self):
        dt = 0.01
        self.phase += 2 * math.pi * self.gait_frequency * dt
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi

        # Left leg
        l_hip, l_knee, l_ankle = self.calculate_leg_joints(self.phase)
        # Right leg (opposite phase)
        r_hip, r_knee, r_ankle = self.calculate_leg_joints(self.phase + self.phase_offset)

        # Publish
        self.publish_joint('left_hip', l_hip)
        self.publish_joint('left_knee', l_knee)
        self.publish_joint('left_ankle', l_ankle)
        self.publish_joint('right_hip', r_hip)
        self.publish_joint('right_knee', r_knee)
        self.publish_joint('right_ankle', r_ankle)

    def calculate_leg_joints(self, phase):
        # Hip: forward/backward swing
        hip = self.hip_amplitude * math.sin(phase)

        # Knee: bend during swing phase
        knee = self.knee_amplitude * max(0, math.sin(phase))

        # Ankle: compensate for terrain
        ankle = -self.ankle_amplitude * math.sin(phase - math.pi/4)

        return hip, knee, ankle

    def publish_joint(self, name, position):
        msg = Float64()
        msg.data = position
        self.pubs[name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    controller = WalkingController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

</details>

---

## Bonus Challenge: Multi-Sensor Fusion

### Objective

Combine IMU and force/torque sensor data to estimate robot state.

### Difficulty

:star::star::star::star: Expert (2+ hours)

### Description

Implement a simple state estimator that fuses:
- IMU orientation and angular velocity
- Foot force/torque sensors for contact detection
- Joint states for kinematic estimation

### Requirements

- Detect stance vs. swing phase for each foot
- Estimate center of mass position
- Detect loss of balance conditions
- Publish estimated state to a topic

### Success Criteria

- [ ] Correctly identifies which foot is in contact
- [ ] Estimates body orientation within 5° accuracy
- [ ] Detects "falling" condition before ground contact
- [ ] Runs at 100Hz minimum

---

## Self-Assessment

After completing these exercises, you should be able to:

- [ ] Create custom Gazebo worlds with obstacles
- [ ] Analyze sensor noise characteristics
- [ ] Implement basic locomotion controllers
- [ ] Debug simulation issues

## Next Steps

- Review the [Chapter 2 Summary](./lesson-05-ros2-gazebo-bridge.md)
- Proceed to [Chapter 3: Perception](../chapter-03-perception/lesson-01-computer-vision.md)
- Experiment with different gait parameters
