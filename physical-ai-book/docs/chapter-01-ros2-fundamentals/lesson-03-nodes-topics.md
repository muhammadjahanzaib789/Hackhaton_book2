---
sidebar_position: 3
title: Nodes and Topics
description: Building ROS 2 publishers and subscribers
---

# Nodes and Topics

## Learning Objectives

By the end of this lesson, you will:

1. Create a ROS 2 Python package from scratch
2. Write a publisher node that sends messages
3. Write a subscriber node that receives messages
4. Understand message types and QoS settings
5. Use command-line tools to inspect topics

## Understanding Nodes

A **node** is the fundamental unit of computation in ROS 2. Each node should do **one thing well**.

### Good Node Design

| Good ✅ | Bad ❌ |
|---------|--------|
| `camera_driver` - publishes images | `robot_brain` - does everything |
| `object_detector` - finds objects | `sensor_processor` - handles all sensors |
| `joint_controller` - controls one joint | `motion_system` - vague responsibility |

### Node Lifecycle

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐
│ Created │ ──▶ │ Inactive │ ──▶ │ Active  │ ──▶ │ Finalized│
└─────────┘     └──────────┘     └─────────┘     └──────────┘
```

For now, we'll use simple nodes that go directly to Active state.

## Understanding Topics

**Topics** are named channels for streaming data. They implement the **publish-subscribe** pattern:

```
┌────────────┐                         ┌────────────┐
│ Publisher  │ ────/sensor_data────▶  │ Subscriber │
│   Node     │                         │    Node    │
└────────────┘                         └────────────┘
                      │
                      ▼
               ┌────────────┐
               │ Subscriber │
               │   Node 2   │
               └────────────┘
```

Key characteristics:
- **One-to-many**: One publisher, many subscribers
- **Asynchronous**: Publisher doesn't wait for subscribers
- **Typed**: Each topic has a specific message type

## Create Your First Package

### Step 1: Create Package Structure

```bash
cd ~/ros2_ws/src

# Create a Python package
ros2 pkg create --build-type ament_python my_first_pkg \
  --dependencies rclpy std_msgs
```

This creates:

```
my_first_pkg/
├── package.xml           # Package metadata
├── setup.py              # Python setup file
├── setup.cfg             # Setup configuration
├── resource/
│   └── my_first_pkg      # Ament resource marker
└── my_first_pkg/         # Python module
    └── __init__.py
```

### Step 2: Understand package.xml

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_first_pkg</name>
  <version>0.0.1</version>
  <description>My first ROS 2 package</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <!-- Dependencies -->
  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Write a Publisher Node

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/simple_publisher.py`:

```python
#!/usr/bin/env python3
"""
Simple Publisher Node
Publishes string messages to the /chatter topic at 1 Hz.

This demonstrates:
- Creating a ROS 2 node
- Creating a publisher
- Using a timer for periodic publishing
- Logging

Expected output:
  [INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 1"
  [INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 2"
  ...
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """A simple publisher node that sends string messages."""

    def __init__(self):
        # Initialize the node with name 'simple_publisher'
        super().__init__('simple_publisher')

        # Create a publisher
        # - Message type: String
        # - Topic name: /chatter
        # - Queue size: 10 (buffer for messages)
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer that calls publish_message every 1 second
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_message)

        # Counter for message numbering
        self.count = 0

        self.get_logger().info('Simple Publisher has started!')

    def publish_message(self):
        """Timer callback - publishes a message."""
        # Create message
        msg = String()
        self.count += 1
        msg.data = f'Hello, ROS 2! Count: {self.count}'

        # Publish
        self.publisher.publish(msg)

        # Log
        self.get_logger().info(f'Publishing: "{msg.data}"')


def main(args=None):
    """Main entry point."""
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create node
    node = SimplePublisher()

    try:
        # Spin (process callbacks)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Write a Subscriber Node

Create `~/ros2_ws/src/my_first_pkg/my_first_pkg/simple_subscriber.py`:

```python
#!/usr/bin/env python3
"""
Simple Subscriber Node
Subscribes to the /chatter topic and logs received messages.

This demonstrates:
- Creating a subscriber
- Callback functions
- Message handling

Expected output:
  [INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 1"
  [INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 2"
  ...
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """A simple subscriber node that receives string messages."""

    def __init__(self):
        # Initialize the node
        super().__init__('simple_subscriber')

        # Create a subscriber
        # - Message type: String
        # - Topic name: /chatter
        # - Callback: self.listener_callback
        # - Queue size: 10
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10
        )

        self.get_logger().info('Simple Subscriber has started!')

    def listener_callback(self, msg):
        """Called when a message is received."""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = SimpleSubscriber()

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

## Register Entry Points

Edit `~/ros2_ws/src/my_first_pkg/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'my_first_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='My first ROS 2 package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = my_first_pkg.simple_publisher:main',
            'simple_subscriber = my_first_pkg.simple_subscriber:main',
        ],
    },
)
```

## Build and Run

### Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select my_first_pkg
source install/setup.bash
```

### Run the Nodes

**Terminal 1** (Publisher):
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_first_pkg simple_publisher
```

**Terminal 2** (Subscriber):
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_first_pkg simple_subscriber
```

### Expected Output

**Terminal 1**:
```
[INFO] [simple_publisher]: Simple Publisher has started!
[INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 1"
[INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Count: 2"
```

**Terminal 2**:
```
[INFO] [simple_subscriber]: Simple Subscriber has started!
[INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 1"
[INFO] [simple_subscriber]: I heard: "Hello, ROS 2! Count: 2"
```

## Command-Line Tools

### List Topics

```bash
ros2 topic list
```

Output:
```
/chatter
/parameter_events
/rosout
```

### Topic Info

```bash
ros2 topic info /chatter
```

Output:
```
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 1
```

### Echo Topic Data

```bash
ros2 topic echo /chatter
```

Output:
```
data: 'Hello, ROS 2! Count: 42'
---
data: 'Hello, ROS 2! Count: 43'
---
```

### Publish from Command Line

```bash
ros2 topic pub /chatter std_msgs/msg/String "{data: 'Hello from CLI'}"
```

### Check Publishing Rate

```bash
ros2 topic hz /chatter
```

Output:
```
average rate: 1.000
    min: 0.999s max: 1.001s std dev: 0.00050s
```

## Message Types

ROS 2 includes many standard message types:

| Package | Common Messages |
|---------|-----------------|
| `std_msgs` | String, Int32, Float64, Bool |
| `geometry_msgs` | Pose, Twist, Point, Vector3 |
| `sensor_msgs` | Image, LaserScan, JointState |
| `nav_msgs` | Odometry, Path, OccupancyGrid |

### View Message Definition

```bash
ros2 interface show std_msgs/msg/String
```

Output:
```
string data
```

### Using geometry_msgs

```python
from geometry_msgs.msg import Twist

# Create velocity command
cmd = Twist()
cmd.linear.x = 0.5   # Forward velocity (m/s)
cmd.angular.z = 0.1  # Rotation velocity (rad/s)
```

## Quality of Service (QoS)

QoS settings control message delivery guarantees:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Reliable delivery (for commands)
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Best effort (for high-frequency sensors)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Use in publisher/subscriber
self.publisher = self.create_publisher(String, 'topic', reliable_qos)
```

## Summary

You've learned to:

- ✅ Create a ROS 2 Python package
- ✅ Write publisher and subscriber nodes
- ✅ Use timers for periodic callbacks
- ✅ Inspect topics with command-line tools
- ✅ Understand message types and QoS

## Exercises

1. Modify the publisher to send `geometry_msgs/msg/Twist` messages
2. Create a node that subscribes and republishes with modifications
3. Experiment with different QoS settings

## Checkpoint

Can you:

1. Create a new package from scratch?
2. Write a publisher that sends custom data?
3. Use `ros2 topic echo` to view messages?

---

**Next**: [Lesson 4: Services and Actions](/docs/chapter-01-ros2-fundamentals/lesson-04-services-actions)
