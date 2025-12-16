---
sidebar_position: 3
title: "Lesson 3: Isaac ROS Perception"
description: "NVIDIA Isaac ROS for accelerated perception pipelines"
---

# Isaac ROS Perception

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand Isaac ROS architecture and benefits
2. Set up depth perception with Isaac ROS
3. Implement 3D object detection
4. Use hardware-accelerated image processing

## Prerequisites

- Completed Lessons 1-2 of this chapter
- NVIDIA GPU (recommended for Isaac ROS)
- Docker installed (Isaac ROS runs in containers)

## What is Isaac ROS?

Isaac ROS is NVIDIA's GPU-accelerated robotics software stack built on ROS 2.

```
┌─────────────────────────────────────────────────────────────┐
│                  Isaac ROS Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   ROS 2 Layer                        │   │
│  │  Standard ROS 2 topics, services, actions           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Isaac ROS Packages                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ Visual   │ │ DNN      │ │ Stereo   │           │   │
│  │  │ SLAM     │ │ Inference│ │ Vision   │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              NVIDIA Hardware                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ CUDA     │ │ TensorRT │ │ VPI      │           │   │
│  │  │ Cores    │ │ (DNN)    │ │ (Vision) │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Isaac ROS Packages

| Package | Purpose | Use Case |
|---------|---------|----------|
| **isaac_ros_visual_slam** | Visual SLAM | Robot localization |
| **isaac_ros_dnn_inference** | DNN inference | Object detection |
| **isaac_ros_depth_image_proc** | Depth processing | Point cloud generation |
| **isaac_ros_stereo_image_proc** | Stereo vision | Depth from stereo |
| **isaac_ros_image_proc** | Image processing | Resize, rectify, convert |
| **isaac_ros_apriltag** | Fiducial detection | Marker-based localization |

## Setting Up Isaac ROS

### Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi

# Install Docker with NVIDIA support
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify NVIDIA container runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Isaac ROS Docker Setup

```bash
# Clone Isaac ROS common
cd ~/workspaces
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Build Isaac ROS development container
cd isaac_ros_common
./scripts/run_dev.sh

# Inside container, build Isaac ROS packages
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

## Depth Perception

### Point Cloud from Depth Image

```python
#!/usr/bin/env python3
"""
Depth to Point Cloud Converter
Physical AI Book - Chapter 3

Converts depth images to 3D point clouds.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
import numpy as np
from cv_bridge import CvBridge


class DepthToPointCloud(Node):
    """Convert depth images to point clouds."""

    def __init__(self):
        super().__init__('depth_to_pointcloud')

        self.bridge = CvBridge()
        self.camera_info = None

        # Subscribers
        self.create_subscription(
            CameraInfo, '/depth_camera/camera_info',
            self.camera_info_callback, 10
        )
        self.create_subscription(
            Image, '/depth_camera/depth',
            self.depth_callback, 10
        )

        # Publisher
        self.pc_pub = self.create_publisher(
            PointCloud2, '/depth_camera/points', 10
        )

        self.get_logger().info('Depth to point cloud ready')

    def camera_info_callback(self, msg):
        """Store camera intrinsics."""
        self.camera_info = msg

    def depth_callback(self, msg):
        """Convert depth image to point cloud."""
        if self.camera_info is None:
            return

        # Convert to numpy
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # Get camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Create point cloud
        height, width = depth.shape
        points = []

        # Create meshgrid for efficiency
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # Calculate 3D coordinates
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        # Filter invalid points (NaN or zero depth)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)
        points = points[valid]

        # Create PointCloud2 message
        pc_msg = point_cloud2.create_cloud_xyz32(msg.header, points)
        self.pc_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Using Isaac ROS Depth Processing

```xml
<!-- Launch file for Isaac ROS depth processing -->
<launch>
  <!-- Depth image to point cloud -->
  <node pkg="isaac_ros_depth_image_proc"
        exec="point_cloud_xyz_node"
        name="point_cloud_xyz">
    <remap from="depth" to="/depth_camera/depth"/>
    <remap from="camera_info" to="/depth_camera/camera_info"/>
    <remap from="points" to="/depth_camera/points"/>
  </node>
</launch>
```

## 3D Object Detection

### Detection with Depth

```python
#!/usr/bin/env python3
"""
3D Object Detection
Physical AI Book - Chapter 3

Combines 2D detection with depth for 3D localization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import numpy as np
import message_filters


class Object3DDetector(Node):
    """Combine 2D detection with depth for 3D localization."""

    def __init__(self):
        super().__init__('object_3d_detector')

        self.bridge = CvBridge()
        self.camera_info = None

        # Camera info
        self.create_subscription(
            CameraInfo, '/depth_camera/camera_info',
            self.camera_info_callback, 10
        )

        # Synchronized subscribers for depth and detections
        depth_sub = message_filters.Subscriber(
            self, Image, '/depth_camera/depth'
        )
        detection_sub = message_filters.Subscriber(
            self, Detection2DArray, '/detections'
        )

        # Time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, detection_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.detection_3d_pub = self.create_publisher(
            Detection3DArray, '/detections_3d', 10
        )

        self.get_logger().info('3D object detector ready')

    def camera_info_callback(self, msg):
        """Store camera intrinsics."""
        self.camera_info = msg

    def sync_callback(self, depth_msg, detection_msg):
        """Process synchronized depth and detections."""
        if self.camera_info is None:
            return

        # Convert depth to numpy
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        # Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Process each detection
        detection_3d_array = Detection3DArray()
        detection_3d_array.header = depth_msg.header

        for det_2d in detection_msg.detections:
            # Get 2D bbox center
            u = int(det_2d.bbox.center.position.x)
            v = int(det_2d.bbox.center.position.y)

            # Get depth at center (use median of region for robustness)
            half_w = int(det_2d.bbox.size_x / 4)
            half_h = int(det_2d.bbox.size_y / 4)

            u_min = max(0, u - half_w)
            u_max = min(depth.shape[1], u + half_w)
            v_min = max(0, v - half_h)
            v_max = min(depth.shape[0], v + half_h)

            depth_region = depth[v_min:v_max, u_min:u_max]
            valid_depths = depth_region[np.isfinite(depth_region) & (depth_region > 0)]

            if len(valid_depths) == 0:
                continue

            z = np.median(valid_depths)

            # Calculate 3D position
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Create 3D detection
            det_3d = Detection3D()
            det_3d.header = depth_msg.header
            det_3d.results = det_2d.results

            # Set 3D pose
            det_3d.bbox.center.position.x = x
            det_3d.bbox.center.position.y = y
            det_3d.bbox.center.position.z = z

            # Estimate 3D size (approximate)
            size_x = det_2d.bbox.size_x * z / fx
            size_y = det_2d.bbox.size_y * z / fy
            det_3d.bbox.size.x = size_x
            det_3d.bbox.size.y = size_y
            det_3d.bbox.size.z = min(size_x, size_y)  # Approximate

            detection_3d_array.detections.append(det_3d)

            # Log
            if det_2d.results:
                class_name = det_2d.results[0].hypothesis.class_id
                self.get_logger().info(
                    f'3D Detection: {class_name} at ({x:.2f}, {y:.2f}, {z:.2f})m'
                )

        self.detection_3d_pub.publish(detection_3d_array)


def main(args=None):
    rclpy.init(args=args)
    node = Object3DDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## AprilTag Detection

AprilTags provide precise pose estimation for manipulation and localization.

```python
#!/usr/bin/env python3
"""
AprilTag Detector
Physical AI Book - Chapter 3

Detects AprilTag fiducial markers for precise localization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np

# Try to import apriltag library
try:
    import apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False


class AprilTagDetector(Node):
    """Detect AprilTag markers and estimate poses."""

    def __init__(self):
        super().__init__('apriltag_detector')

        if not APRILTAG_AVAILABLE:
            self.get_logger().error('apriltag library not available')
            self.get_logger().error('Install with: pip install apriltag')
            return

        self.bridge = CvBridge()
        self.camera_info = None

        # AprilTag detector options
        options = apriltag.DetectorOptions(families='tag36h11')
        self.detector = apriltag.Detector(options)

        # Tag size in meters
        self.declare_parameter('tag_size', 0.1)
        self.tag_size = self.get_parameter('tag_size').value

        # Subscribers
        self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.camera_info_callback, 10
        )
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseArray, '/apriltag/poses', 10
        )
        self.debug_pub = self.create_publisher(
            Image, '/apriltag/debug', 10
        )

        self.get_logger().info('AprilTag detector ready')

    def camera_info_callback(self, msg):
        """Store camera intrinsics."""
        self.camera_info = msg

    def image_callback(self, msg):
        """Detect AprilTags in image."""
        if self.camera_info is None:
            return

        # Convert to grayscale
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect tags
        detections = self.detector.detect(gray)

        # Get camera parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        camera_params = (fx, fy, cx, cy)

        # Process detections
        pose_array = PoseArray()
        pose_array.header = msg.header

        debug_image = cv_image.copy()

        for det in detections:
            # Estimate pose
            pose, e0, e1 = self.detector.detection_pose(
                det, camera_params, self.tag_size
            )

            # Extract position and rotation
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]

            # Convert rotation matrix to quaternion
            quat = self.rotation_to_quaternion(rotation_matrix)

            # Create Pose message
            tag_pose = Pose()
            tag_pose.position.x = float(position[0])
            tag_pose.position.y = float(position[1])
            tag_pose.position.z = float(position[2])
            tag_pose.orientation.x = quat[0]
            tag_pose.orientation.y = quat[1]
            tag_pose.orientation.z = quat[2]
            tag_pose.orientation.w = quat[3]

            pose_array.poses.append(tag_pose)

            # Draw on debug image
            corners = det.corners.astype(int)
            cv2.polylines(debug_image, [corners], True, (0, 255, 0), 2)
            center = tuple(det.center.astype(int))
            cv2.circle(debug_image, center, 5, (0, 0, 255), -1)
            cv2.putText(
                debug_image, f'ID: {det.tag_id}',
                (center[0] - 20, center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

            self.get_logger().info(
                f'Tag {det.tag_id}: pos=({position[0]:.3f}, '
                f'{position[1]:.3f}, {position[2]:.3f})'
            )

        self.pose_pub.publish(pose_array)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
        self.debug_pub.publish(debug_msg)

    def rotation_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return [x, y, z, w]


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **Isaac ROS** provides GPU-accelerated perception
2. **Depth images** enable 3D understanding
3. **3D detection** combines 2D detection with depth
4. **AprilTags** provide precise pose estimation
5. **Hardware acceleration** is critical for real-time performance

## Next Steps

Continue to [Chapter 4: Navigation](../chapter-04-navigation/lesson-01-slam.md) to learn:
- SLAM for robot localization
- Path planning algorithms
- Nav2 integration

## Additional Resources

- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [Point Cloud Library (PCL)](https://pointclouds.org/)
- [AprilTag Documentation](https://april.eecs.umich.edu/software/apriltag)
