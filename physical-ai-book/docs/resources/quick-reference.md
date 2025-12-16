---
sidebar_position: 3
title: "Quick Reference"
description: "Essential commands and code patterns for Physical AI development"
---

# Quick Reference

Essential commands, patterns, and code snippets for Physical AI robotics development.

## ROS 2 Commands

### Workspace Management

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build
colcon build --packages-select my_package

# Source workspace
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Package Management

```bash
# Create Python package
ros2 pkg create --build-type ament_python my_package

# Create C++ package
ros2 pkg create --build-type ament_cmake my_package --dependencies rclcpp

# List packages
ros2 pkg list
ros2 pkg prefix my_package
```

### Node Operations

```bash
# Run a node
ros2 run package_name node_name

# List running nodes
ros2 node list
ros2 node info /node_name

# Remap topic
ros2 run pkg node --ros-args -r /old_topic:=/new_topic

# Set parameters
ros2 run pkg node --ros-args -p param:=value
```

### Topic Operations

```bash
# List topics
ros2 topic list
ros2 topic list -t  # Show types

# Topic info
ros2 topic info /topic_name
ros2 topic hz /topic_name  # Frequency
ros2 topic bw /topic_name  # Bandwidth

# Publish/Subscribe
ros2 topic pub /topic type '{data}'
ros2 topic echo /topic_name
```

### Service Operations

```bash
# List services
ros2 service list
ros2 service type /service_name

# Call service
ros2 service call /service_name type '{request}'
```

### Action Operations

```bash
# List actions
ros2 action list
ros2 action info /action_name

# Send goal
ros2 action send_goal /action_name type '{goal}'
```

### Launch Files

```bash
# Run launch file
ros2 launch package_name launch_file.py
ros2 launch package_name file.launch.py arg:=value
```

---

## Common ROS 2 Patterns

### Python Node Template

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Parameters
        self.declare_parameter('param_name', 'default')
        self.param = self.get_parameter('param_name').value

        # Publisher
        self.pub = self.create_publisher(String, '/topic', 10)

        # Subscriber
        self.sub = self.create_subscription(
            String, '/other_topic',
            self.callback, 10
        )

        # Timer
        self.timer = self.create_timer(1.0, self.timer_callback)

    def callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello'
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Server Template

```python
from example_interfaces.srv import AddTwoInts

class MyService(Node):
    def __init__(self):
        super().__init__('my_service')
        self.srv = self.create_service(
            AddTwoInts, 'add_two_ints',
            self.callback
        )

    def callback(self, request, response):
        response.sum = request.a + request.b
        return response
```

### Action Server Template

```python
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class MyActionServer(Node):
    def __init__(self):
        super().__init__('my_action_server')
        self._action_server = ActionServer(
            self, Fibonacci, 'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        result = Fibonacci.Result()
        # Do work...
        goal_handle.succeed()
        return result
```

---

## Gazebo Commands

```bash
# Launch Gazebo
ros2 launch gazebo_ros gazebo.launch.py

# Spawn robot
ros2 run gazebo_ros spawn_entity.py \
    -topic robot_description \
    -entity robot_name

# World control
ros2 service call /world_control gazebo_msgs/srv/SetPhysicsProperties
```

---

## Nav2 Commands

```bash
# Launch Nav2
ros2 launch nav2_bringup navigation_launch.py

# Launch SLAM
ros2 launch slam_toolbox online_async_launch.py

# Send navigation goal
ros2 action send_goal /navigate_to_pose \
    nav2_msgs/action/NavigateToPose \
    "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.0}}}}"

# Save map
ros2 run nav2_map_server map_saver_cli -f my_map
```

---

## MoveIt2 Commands

```bash
# Launch MoveIt2
ros2 launch my_robot_moveit demo.launch.py

# Plan and execute via Python
from moveit_py import MoveItPy

moveit = MoveItPy(node_name="moveit_py")
arm = moveit.get_planning_component("arm")
arm.set_goal_state(configuration_name="home")
arm.plan()
arm.execute()
```

---

## TF2 Commands

```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo transform
ros2 run tf2_ros tf2_echo frame1 frame2

# Static transform publisher
ros2 run tf2_ros static_transform_publisher \
    x y z yaw pitch roll parent_frame child_frame
```

---

## Common Message Types

### Geometry Messages

```python
from geometry_msgs.msg import (
    Point,          # x, y, z
    Pose,           # position, orientation
    PoseStamped,    # header, pose
    Twist,          # linear, angular
    Transform,      # translation, rotation
    Quaternion,     # x, y, z, w
)

# Create Twist (velocity command)
twist = Twist()
twist.linear.x = 0.5
twist.angular.z = 0.1

# Create PoseStamped
pose = PoseStamped()
pose.header.frame_id = 'map'
pose.pose.position.x = 1.0
pose.pose.position.y = 2.0
pose.pose.orientation.w = 1.0
```

### Sensor Messages

```python
from sensor_msgs.msg import (
    Image,           # Camera image
    LaserScan,       # 2D LiDAR
    PointCloud2,     # 3D point cloud
    Imu,             # IMU data
    JointState,      # Joint positions/velocities
    BatteryState,    # Battery info
)
```

### Navigation Messages

```python
from nav_msgs.msg import (
    Odometry,        # Robot odometry
    Path,            # Planned path
    OccupancyGrid,   # 2D map
)
```

---

## Image Processing

### CV Bridge Pattern

```python
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def image_callback(msg):
    # ROS to OpenCV
    cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')

    # Process with OpenCV
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # OpenCV to ROS
    ros_image = bridge.cv2_to_imgmsg(gray, 'mono8')
```

### Depth Image Processing

```python
def depth_callback(msg):
    # Convert to numpy array
    depth = bridge.imgmsg_to_cv2(msg, 'passthrough')

    # Get depth at pixel (u, v)
    depth_value = depth[v, u]  # in meters
```

---

## Common Coordinate Transforms

### Euler to Quaternion

```python
from scipy.spatial.transform import Rotation
import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()  # [x, y, z, w]

def quaternion_to_euler(x, y, z, w):
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler('xyz')  # [roll, pitch, yaw]
```

### Transform Point Between Frames

```python
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)

# Get transform
transform = tf_buffer.lookup_transform(
    'target_frame', 'source_frame',
    rclpy.time.Time()
)

# Transform point
transformed_point = do_transform_point(point_stamped, transform)
```

---

## LLM Integration

### Ollama Quick Start

```python
import aiohttp
import json

async def query_ollama(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:11434/api/generate',
            json={'model': 'llama3.2', 'prompt': prompt}
        ) as resp:
            result = await resp.json()
            return result['response']
```

### Action Schema

```python
ACTION_SCHEMA = {
    "navigate": {
        "params": ["target_x", "target_y", "target_yaw"],
        "description": "Move robot to position"
    },
    "pick": {
        "params": ["object_name"],
        "description": "Pick up an object"
    },
    "speak": {
        "params": ["message"],
        "description": "Say something"
    }
}
```

---

## Testing Commands

```bash
# Run all tests
colcon test
colcon test-result --verbose

# Run specific test
colcon test --packages-select my_package

# Pytest directly
pytest tests/ -v

# Launch integration tests
ros2 launch my_package test_launch.py
```

---

## Debugging

### Logger Levels

```python
self.get_logger().debug('Debug message')
self.get_logger().info('Info message')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
```

### RQt Tools

```bash
# Plot topics
ros2 run rqt_plot rqt_plot

# Image view
ros2 run rqt_image_view rqt_image_view

# TF tree
ros2 run rqt_tf_tree rqt_tf_tree

# Node graph
ros2 run rqt_graph rqt_graph

# Console
ros2 run rqt_console rqt_console
```

### Bag Recording

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /topic1 /topic2

# Play bag
ros2 bag play my_bag

# Bag info
ros2 bag info my_bag
```

---

## Environment Variables

```bash
# ROS 2 domain
export ROS_DOMAIN_ID=42

# Middleware
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Localhost only
export ROS_LOCALHOST_ONLY=1

# Log level
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_COLORIZED_OUTPUT=1
```

---

## Docker Quick Start

```dockerfile
FROM ros:humble

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-nav2-* \
    ros-humble-moveit \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace
COPY . /ros2_ws/src

# Build
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build

# Entry point
COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

```bash
# Build and run
docker build -t my_robot .
docker run -it --rm my_robot
```
