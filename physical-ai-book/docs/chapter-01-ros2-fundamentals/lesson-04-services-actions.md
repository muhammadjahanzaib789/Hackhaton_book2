---
sidebar_position: 4
title: Services and Actions
description: Request-response and long-running task patterns in ROS 2
---

# Services and Actions

## Learning Objectives

By the end of this lesson, you will:

1. Understand when to use services vs topics vs actions
2. Create a service server and client
3. Create an action server and client
4. Handle feedback and cancellation in actions

## Choosing the Right Pattern

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Topic** | Continuous data stream | Camera images, sensor readings |
| **Service** | Quick request/response | Get robot state, set parameter |
| **Action** | Long-running task | Navigate to goal, pick up object |

### Decision Flowchart

```
Is the data continuous/streaming?
├─ YES → Use Topic
└─ NO → Does it complete quickly (<1 second)?
         ├─ YES → Use Service
         └─ NO → Do you need feedback/cancellation?
                  ├─ YES → Use Action
                  └─ NO → Use Service (but consider Action)
```

## Services: Request/Response

Services follow a **synchronous** request/response pattern:

```
┌────────────┐    Request          ┌────────────┐
│   Client   │ ─────────────────▶  │   Server   │
│            │ ◀─────────────────  │            │
└────────────┘    Response         └────────────┘
```

### Service Definition

Services use `.srv` files with request and response sections:

```
# Example: AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

### Create a Service Server

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/add_service.py`:

```python
#!/usr/bin/env python3
"""
Service Server Example
Provides an 'add_two_ints' service that adds two integers.

Expected usage:
  ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 3}"

Expected response:
  sum: 8
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddService(Node):
    """Service server that adds two integers."""

    def __init__(self):
        super().__init__('add_service')

        # Create service
        self.srv = self.create_service(
            AddTwoInts,           # Service type
            'add_two_ints',       # Service name
            self.add_callback     # Callback function
        )

        self.get_logger().info('Add Service ready!')

    def add_callback(self, request, response):
        """Handle service request."""
        response.sum = request.a + request.b
        self.get_logger().info(
            f'Request: {request.a} + {request.b} = {response.sum}'
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AddService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Create a Service Client

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/add_client.py`:

```python
#!/usr/bin/env python3
"""
Service Client Example
Calls the 'add_two_ints' service.

Expected output:
  [INFO] [add_client]: Result: 5 + 3 = 8
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddClient(Node):
    """Service client that requests addition."""

    def __init__(self):
        super().__init__('add_client')

        # Create client
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

    def send_request(self, a, b):
        """Send request and wait for response."""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Call service (synchronous)
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()


def main(args=None):
    rclpy.init(args=args)

    client = AddClient()

    # Get numbers from command line or use defaults
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    response = client.send_request(a, b)
    client.get_logger().info(f'Result: {a} + {b} = {response.sum}')

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Command-Line Tools

```bash
# List services
ros2 service list

# Get service type
ros2 service type /add_two_ints

# Call service from CLI
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 10, b: 20}"
```

## Actions: Long-Running Tasks

Actions are for tasks that:
- Take a long time (seconds to minutes)
- Need progress feedback
- Should be cancellable

```
┌────────────┐    Goal             ┌────────────┐
│   Client   │ ─────────────────▶  │   Server   │
│            │ ◀── Feedback ────   │            │
│            │ ◀── Feedback ────   │            │
│            │ ◀── Feedback ────   │            │
│            │ ◀── Result ───────  │            │
└────────────┘                     └────────────┘
```

### Action Definition

Actions use `.action` files with goal, result, and feedback:

```
# Example: Fibonacci.action
int32 order          # Goal
---
int32[] sequence     # Result
---
int32[] partial      # Feedback
```

### Create an Action Server

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/countdown_server.py`:

```python
#!/usr/bin/env python3
"""
Action Server Example
Provides a countdown action that counts down from a given number.

This demonstrates:
- Creating an action server
- Publishing feedback
- Handling cancellation
- Returning results

Expected behavior:
  Goal: count_from = 5
  Feedback: 5, 4, 3, 2, 1
  Result: success = True, final_count = 0
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from example_interfaces.action import Fibonacci  # We'll reuse this


class CountdownServer(Node):
    """Action server that performs a countdown."""

    def __init__(self):
        super().__init__('countdown_server')

        self._action_server = ActionServer(
            self,
            Fibonacci,  # Reusing Fibonacci action type
            'countdown',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Countdown Action Server ready!')

    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info(f'Received goal: count from {goal_request.order}')

        # Accept all positive goals
        if goal_request.order > 0:
            return GoalResponse.ACCEPT
        else:
            self.get_logger().warn('Rejecting goal: must be positive')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancellation request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle):
        """Execute the countdown."""
        self.get_logger().info('Executing countdown...')

        feedback_msg = Fibonacci.Feedback()
        result = Fibonacci.Result()

        current = goal_handle.request.order
        sequence = []

        while current > 0:
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled!')
                result.sequence = sequence
                return result

            # Count down
            sequence.append(current)
            feedback_msg.partial_sequence = sequence

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {current}')

            current -= 1
            time.sleep(1.0)  # 1 second per count

        # Success!
        goal_handle.succeed()
        result.sequence = sequence + [0]  # Include final 0
        self.get_logger().info('Countdown complete!')

        return result


def main(args=None):
    rclpy.init(args=args)
    node = CountdownServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Create an Action Client

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/countdown_client.py`:

```python
#!/usr/bin/env python3
"""
Action Client Example
Sends a goal to the countdown action server.

Expected output:
  [INFO] Goal accepted
  [INFO] Feedback: [5]
  [INFO] Feedback: [5, 4]
  [INFO] Feedback: [5, 4, 3]
  [INFO] Feedback: [5, 4, 3, 2]
  [INFO] Feedback: [5, 4, 3, 2, 1]
  [INFO] Result: [5, 4, 3, 2, 1, 0]
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from example_interfaces.action import Fibonacci


class CountdownClient(Node):
    """Action client for countdown."""

    def __init__(self):
        super().__init__('countdown_client')

        self._action_client = ActionClient(
            self,
            Fibonacci,
            'countdown'
        )

    def send_goal(self, count_from):
        """Send a countdown goal."""
        goal_msg = Fibonacci.Goal()
        goal_msg.order = count_from

        self.get_logger().info(f'Sending goal: count from {count_from}')

        # Wait for server
        self._action_client.wait_for_server()

        # Send goal with feedback callback
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Called when goal is accepted/rejected."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return

        self.get_logger().info('Goal accepted!')

        # Request result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Called when feedback is received."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        """Called when result is received."""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

        # Shutdown after receiving result
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    client = CountdownClient()
    client.send_goal(5)  # Count down from 5

    rclpy.spin(client)


if __name__ == '__main__':
    main()
```

### Action Command-Line Tools

```bash
# List actions
ros2 action list

# Get action info
ros2 action info /countdown

# Send goal from CLI
ros2 action send_goal /countdown example_interfaces/action/Fibonacci "{order: 5}"

# Send goal with feedback
ros2 action send_goal /countdown example_interfaces/action/Fibonacci "{order: 5}" --feedback
```

## Update setup.py

Add new entry points:

```python
entry_points={
    'console_scripts': [
        'simple_publisher = my_first_pkg.simple_publisher:main',
        'simple_subscriber = my_first_pkg.simple_subscriber:main',
        'add_service = my_first_pkg.add_service:main',
        'add_client = my_first_pkg.add_client:main',
        'countdown_server = my_first_pkg.countdown_server:main',
        'countdown_client = my_first_pkg.countdown_client:main',
    ],
},
```

## Build and Test

```bash
cd ~/ros2_ws
colcon build --packages-select my_first_pkg
source install/setup.bash

# Test service (two terminals)
# Terminal 1:
ros2 run my_first_pkg add_service
# Terminal 2:
ros2 run my_first_pkg add_client 10 20

# Test action (two terminals)
# Terminal 1:
ros2 run my_first_pkg countdown_server
# Terminal 2:
ros2 run my_first_pkg countdown_client
```

## Real-World Examples

### Service: Get Robot State

```python
# Service definition (pseudo-code)
# GetRobotState.srv
---
float64 battery_level
string current_task
bool is_moving
```

### Action: Navigate to Pose

```python
# Action definition (pseudo-code)
# NavigateToPose.action
geometry_msgs/PoseStamped goal_pose
---
bool success
string message
---
geometry_msgs/PoseStamped current_pose
float32 distance_remaining
float32 eta_seconds
```

This is exactly how Nav2 navigation works!

## Summary

| Pattern | Blocking? | Feedback? | Cancel? | Best For |
|---------|-----------|-----------|---------|----------|
| Topic | No | N/A | N/A | Continuous data |
| Service | Yes | No | No | Quick queries |
| Action | No | Yes | Yes | Long tasks |

You've learned:

- ✅ When to use services vs actions
- ✅ How to create service servers and clients
- ✅ How to create action servers and clients
- ✅ How to handle feedback and cancellation

## Checkpoint

Can you:

1. Explain when to use a service vs an action?
2. Create a service that returns robot status?
3. Create an action that moves a robot arm?

---

**Next**: [Lesson 5: URDF Basics](/docs/chapter-01-ros2-fundamentals/lesson-05-urdf-basics)
