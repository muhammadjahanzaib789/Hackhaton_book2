---
sidebar_position: 6
title: Exercises
description: Hands-on practice for ROS 2 fundamentals
---

# Chapter 1 Exercises

## Overview

These exercises reinforce the concepts from Chapter 1. Complete them in order, as each builds on the previous.

**Estimated time**: 2-3 hours

## Exercise 1: Custom Publisher/Subscriber

**Difficulty**: Easy | **Time**: 20 minutes

### Task

Create a publisher that sends `geometry_msgs/msg/Twist` messages representing robot velocity commands, and a subscriber that receives and logs them.

### Requirements

1. Publisher node named `velocity_commander`
2. Publishes to topic `/cmd_vel`
3. Sends velocity commands at 10 Hz
4. Subscriber node named `velocity_monitor`
5. Logs received linear and angular velocities

### Expected Output

**Publisher**:
```
[INFO] [velocity_commander]: Sending: linear=0.5, angular=0.1
[INFO] [velocity_commander]: Sending: linear=0.5, angular=0.1
```

**Subscriber**:
```
[INFO] [velocity_monitor]: Received: linear.x=0.5, angular.z=0.1
```

### Hints

<details>
<summary>Click for hints</summary>

```python
from geometry_msgs.msg import Twist

# Create message
msg = Twist()
msg.linear.x = 0.5    # Forward velocity (m/s)
msg.linear.y = 0.0
msg.linear.z = 0.0
msg.angular.x = 0.0
msg.angular.y = 0.0
msg.angular.z = 0.1   # Rotation velocity (rad/s)
```

</details>

### Verification

```bash
# In separate terminals:
ros2 run my_first_pkg velocity_commander
ros2 run my_first_pkg velocity_monitor
ros2 topic echo /cmd_vel
ros2 topic hz /cmd_vel  # Should show ~10 Hz
```

---

## Exercise 2: Parameter Service

**Difficulty**: Medium | **Time**: 30 minutes

### Task

Create a service that allows getting and setting robot parameters (speed limit, name, etc.).

### Requirements

1. Service server named `param_server`
2. Custom service type or use `example_interfaces/srv/SetBool`
3. Service `/set_speed_limit` that accepts a boolean (True = fast, False = slow)
4. Returns the new speed limit value

### Expected Behavior

```bash
# Set to fast mode
ros2 service call /set_speed_limit example_interfaces/srv/SetBool "{data: true}"
# Response: success=True, message="Speed limit set to 2.0 m/s"

# Set to slow mode
ros2 service call /set_speed_limit example_interfaces/srv/SetBool "{data: false}"
# Response: success=True, message="Speed limit set to 0.5 m/s"
```

### Hints

<details>
<summary>Click for hints</summary>

```python
from example_interfaces.srv import SetBool

def set_speed_callback(self, request, response):
    if request.data:
        self.speed_limit = 2.0
        response.message = "Speed limit set to 2.0 m/s"
    else:
        self.speed_limit = 0.5
        response.message = "Speed limit set to 0.5 m/s"
    response.success = True
    return response
```

</details>

---

## Exercise 3: Timer Action

**Difficulty**: Medium | **Time**: 45 minutes

### Task

Create an action server that acts as a countdown timer, providing feedback every second.

### Requirements

1. Action server named `timer_server`
2. Action name: `/timer`
3. Goal: duration in seconds (integer)
4. Feedback: seconds remaining
5. Result: total time elapsed, success status
6. Support cancellation

### Expected Behavior

```bash
# Start 10-second timer
ros2 action send_goal /timer example_interfaces/action/Fibonacci "{order: 10}" --feedback

# Output:
# Feedback: 10 seconds remaining
# Feedback: 9 seconds remaining
# ...
# Feedback: 1 seconds remaining
# Result: sequence=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

### Hints

<details>
<summary>Click for hints</summary>

- Reuse the `Fibonacci` action type (order = seconds)
- Use `time.sleep(1.0)` for timing
- Check `goal_handle.is_cancel_requested` each iteration
- Store countdown values in the `partial_sequence` feedback field

</details>

---

## Exercise 4: Multi-Node System

**Difficulty**: Hard | **Time**: 60 minutes

### Task

Create a complete system with multiple nodes communicating via topics, services, and actions.

### System Architecture

```
┌─────────────────┐          ┌─────────────────┐
│   Commander     │ ────────▶│    Controller   │
│ (Publishes      │  /cmd    │ (Processes      │
│  commands)      │          │  commands)      │
└─────────────────┘          └────────┬────────┘
                                      │
                                      ▼ /status
┌─────────────────┐          ┌─────────────────┐
│    Monitor      │ ◀────────│    Controller   │
│ (Displays       │          │ (Publishes      │
│  status)        │          │  status)        │
└─────────────────┘          └─────────────────┘
```

### Requirements

1. **Commander Node**:
   - Publishes `String` commands to `/cmd` at 1 Hz
   - Commands: "forward", "backward", "stop", "turn_left", "turn_right"

2. **Controller Node**:
   - Subscribes to `/cmd`
   - Maintains internal state (position, direction)
   - Publishes status to `/status` as JSON string
   - Provides service `/get_state` returning current position

3. **Monitor Node**:
   - Subscribes to `/status`
   - Logs status changes
   - Calls `/get_state` service every 5 seconds

### Expected Output

**Controller**:
```
[INFO] [controller]: Received command: forward
[INFO] [controller]: Position: (1, 0), Direction: NORTH
```

**Monitor**:
```
[INFO] [monitor]: Status: {"x": 1, "y": 0, "direction": "NORTH"}
[INFO] [monitor]: State query: x=1, y=0
```

### Hints

<details>
<summary>Click for hints</summary>

```python
import json

# Controller state
self.x = 0
self.y = 0
self.direction = "NORTH"  # NORTH, SOUTH, EAST, WEST

# Process command
def process_command(self, cmd):
    if cmd == "forward":
        if self.direction == "NORTH":
            self.y += 1
        elif self.direction == "SOUTH":
            self.y -= 1
        # ... etc

# Publish status as JSON
status = json.dumps({
    "x": self.x,
    "y": self.y,
    "direction": self.direction
})
```

</details>

---

## Exercise 5: URDF Exploration

**Difficulty**: Easy | **Time**: 20 minutes

### Task

Explore the book's humanoid URDF and answer questions.

### Steps

1. Open `static/models/humanoid/humanoid.urdf`
2. Examine the structure

### Questions

Answer these questions by reading the URDF:

1. **Links**: How many links does the humanoid have?

2. **Joint Count**: Verify there are 21 movable joints. List them by body part.

3. **Joint Limits**: What is the range of motion (in degrees) for:
   - `r_elbow` joint
   - `r_knee` joint
   - `waist_yaw` joint

4. **Mass**: What is the total mass of the robot? (Sum all link masses)

5. **Coordinate System**: Which direction does the robot face when all joints are at 0?

### Verification

<details>
<summary>Click for answers</summary>

1. **Links**: 27 links (including intermediate links for compound joints)

2. **Joints by body part**:
   - Torso: waist_yaw (1)
   - Head: neck_yaw (1)
   - Right Arm: r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw, r_elbow, r_wrist_pitch, r_wrist_roll, r_wrist_yaw (7)
   - Left Arm: Same as right (7)
   - Right Leg: r_hip_pitch, r_hip_roll, r_hip_yaw, r_knee, r_ankle_pitch, r_ankle_roll (6)
   - Left Leg: Same as right (6)
   - Total: 1 + 7 + 7 + 6 + 6 = 21 (excluding neck_yaw for camera)

3. **Joint limits** (radians → degrees):
   - r_elbow: 0 to 2.5 rad = 0° to 143°
   - r_knee: 0 to 2.5 rad = 0° to 143°
   - waist_yaw: -1.57 to 1.57 rad = -90° to 90°

4. **Total mass**: ~31.7 kg
   - base_link: 5.0, torso: 10.0, head: 2.0
   - Arms: 2 × (1.5 + 1.0 + 0.3) = 5.6
   - Legs: 2 × (3.0 + 2.0 + 0.5) = 11.0

5. **Facing**: Positive X direction (forward)

</details>

---

## Bonus Challenge: Launch File

**Difficulty**: Hard | **Time**: 30 minutes

### Task

Create a launch file that starts all nodes from Exercise 4 together.

### Requirements

1. Create `launch/multi_node.launch.py`
2. Start commander, controller, and monitor nodes
3. Add arguments for command rate and monitor interval
4. Include proper shutdown handling

### Expected Usage

```bash
ros2 launch my_first_pkg multi_node.launch.py cmd_rate:=2.0
```

### Hints

<details>
<summary>Click for hints</summary>

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('cmd_rate', default_value='1.0'),

        Node(
            package='my_first_pkg',
            executable='commander',
            parameters=[{
                'rate': LaunchConfiguration('cmd_rate')
            }]
        ),
        # ... more nodes
    ])
```

</details>

---

## Submission Checklist

Before moving to Chapter 2, verify:

- [ ] Exercise 1: Velocity publisher/subscriber working at 10 Hz
- [ ] Exercise 2: Parameter service responds correctly
- [ ] Exercise 3: Timer action provides feedback and supports cancel
- [ ] Exercise 4: Multi-node system communicates correctly
- [ ] Exercise 5: URDF questions answered correctly
- [ ] Bonus: Launch file starts all nodes (optional)

## Common Issues

### "Package not found"

```bash
source ~/ros2_ws/install/setup.bash
```

### "Cannot import module"

Rebuild after adding new files:
```bash
colcon build --packages-select my_first_pkg
source install/setup.bash
```

### "Service not available"

Ensure server is running before client:
```bash
ros2 service list  # Check if service exists
```

---

**Congratulations!** You've completed Chapter 1: ROS 2 Fundamentals.

**Next**: [Chapter 2: Simulation](/docs/chapter-02-simulation)
