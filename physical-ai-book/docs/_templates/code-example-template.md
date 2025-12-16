---
sidebar_position: 99
title: "Code Example: [Name]"
description: "Complete working example for [topic]"
---

# Code Example: [Name]

## Overview

Brief description of what this code example demonstrates.

**Source File**: `src/examples/ros2/chXX/example_name.py`

## Prerequisites

- ROS 2 Humble installed
- Python 3.10+
- Required packages: `package1`, `package2`

## Installation

```bash
# Install dependencies
sudo apt install ros-humble-package-name

# Build the examples
cd physical-ai-book
colcon build --packages-select physical_ai_examples
source install/setup.bash
```

## Complete Code

```python
#!/usr/bin/env python3
"""
[Example Name]
Physical AI Book - Chapter X: [Chapter Name]

[Detailed description of what this example does]

Usage:
    ros2 run physical_ai_examples example_name

Expected Output:
    [INFO] [node_name]: Expected output line 1
    [INFO] [node_name]: Expected output line 2

Dependencies:
    - rclpy
    - std_msgs
    - [other dependencies]

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node


class ExampleNode(Node):
    """
    [Class description]

    Publishes:
        /topic_name (MessageType): Description

    Subscribes:
        /input_topic (MessageType): Description

    Services:
        /service_name (ServiceType): Description

    Parameters:
        param_name (type): Description [default: value]
    """

    def __init__(self):
        super().__init__('node_name')

        # === Parameters ===
        self.declare_parameter('param_name', 'default_value')
        self.param_value = self.get_parameter('param_name').value

        # === Publishers ===
        self.publisher = self.create_publisher(
            MessageType,
            'topic_name',
            10  # QoS depth
        )

        # === Subscribers ===
        self.subscription = self.create_subscription(
            MessageType,
            'input_topic',
            self.callback,
            10
        )

        # === Timers ===
        self.timer = self.create_timer(1.0, self.timer_callback)

        # === State ===
        self.counter = 0

        self.get_logger().info('Node initialized')

    def callback(self, msg):
        """
        Callback for incoming messages.

        Args:
            msg: The received message
        """
        self.get_logger().info(f'Received: {msg.data}')

    def timer_callback(self):
        """Periodic callback for publishing."""
        self.counter += 1
        # Publish logic here


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = ExampleNode()

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

## Running the Example

### Terminal 1: Start the Node

```bash
ros2 run physical_ai_examples example_name
```

### Terminal 2: Interact with the Node

```bash
# Publish to input topic
ros2 topic pub /input_topic std_msgs/msg/String "data: 'Hello'"

# Monitor output topic
ros2 topic echo /topic_name
```

## Expected Output

```
[INFO] [example_node]: Node initialized
[INFO] [example_node]: Received: Hello
[INFO] [example_node]: Published: Response
```

## Code Walkthrough

### Initialization

The `__init__` method sets up:
1. **Parameters**: Configurable values that can be set at runtime
2. **Publishers**: Outgoing message channels
3. **Subscribers**: Incoming message channels
4. **Timers**: Periodic callbacks

### Message Flow

```
Input Topic ──▶ Callback ──▶ Processing ──▶ Publisher ──▶ Output Topic
```

### Key Points

- Point 1 about the implementation
- Point 2 about design decisions
- Point 3 about best practices

## Modifications

### Customization 1

How to modify the code for a different use case:

```python
# Modified code snippet
```

### Customization 2

Alternative approach for specific scenarios:

```python
# Alternative code snippet
```

## Troubleshooting

### Issue 1: Node doesn't start

**Cause**: Missing dependency or wrong Python version

**Solution**:
```bash
pip install missing-package
```

### Issue 2: Messages not received

**Cause**: QoS mismatch between publisher and subscriber

**Solution**: Ensure QoS profiles match

## Related Examples

- [Previous Example](./previous-example.md)
- [Next Example](./next-example.md)
- [Related Concept](../lesson-xx.md)
