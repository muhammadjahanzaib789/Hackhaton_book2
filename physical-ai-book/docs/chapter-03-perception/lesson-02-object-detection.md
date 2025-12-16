---
sidebar_position: 2
title: "Lesson 2: Object Detection"
description: "Deep learning-based object detection with YOLO"
---

# Object Detection with Deep Learning

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand object detection architectures
2. Run YOLO inference in ROS 2
3. Process detection results for robotics
4. Handle multiple object classes

## Prerequisites

- Completed Lesson 1 (Computer Vision Fundamentals)
- Basic understanding of neural networks
- GPU recommended but not required

## Object Detection Overview

Object detection combines classification (what?) with localization (where?).

```
┌─────────────────────────────────────────────────────────────┐
│              Object Detection Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image ──▶ Neural ──▶ Post-      ──▶ Detections            │
│  Input     Network    Processing      Output                │
│                                                             │
│  ┌───────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │       │  │  Backbone      │  │ class: "cup"           │ │
│  │ 640×  │─▶│  (feature      │─▶│ confidence: 0.95       │ │
│  │ 480   │  │   extraction)  │  │ bbox: [100,200,50,80]  │ │
│  │       │  │       ↓        │  │                        │ │
│  │       │  │  Detection     │  │ class: "bottle"        │ │
│  │       │  │  Head          │  │ confidence: 0.87       │ │
│  └───────┘  └────────────────┘  │ bbox: [300,150,40,120] │ │
│                                  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Detection vs. Classification

| Task | Output | Use Case |
|------|--------|----------|
| Classification | Class label | "This is a cup" |
| Detection | Class + bounding box | "Cup at (100, 200)" |
| Segmentation | Class + pixel mask | Precise object boundary |
| Pose Estimation | Keypoints | Object orientation |

## YOLO Architecture

YOLO (You Only Look Once) is a popular real-time detector.

### YOLO Versions

| Version | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| YOLOv5n | Fastest | Good | Edge devices |
| YOLOv5s | Fast | Better | Real-time robotics |
| YOLOv5m | Medium | High | Balanced |
| YOLOv8n | Fastest | Good | Latest architecture |
| YOLOv8s | Fast | Better | Recommended |

### Installing Ultralytics YOLO

```bash
# Install ultralytics package
pip install ultralytics

# Verify installation
python3 -c "from ultralytics import YOLO; print('YOLO ready')"
```

## Basic YOLO Inference

```python
#!/usr/bin/env python3
"""
YOLO Object Detection
Physical AI Book - Chapter 3

Basic YOLO inference on images.
"""

from ultralytics import YOLO
import cv2


def detect_objects(image_path, model_name='yolov8n.pt'):
    """
    Run YOLO detection on an image.

    Args:
        image_path: Path to image file
        model_name: YOLO model to use

    Returns:
        Detection results
    """
    # Load model (downloads automatically if needed)
    model = YOLO(model_name)

    # Run inference
    results = model(image_path)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Confidence score
            confidence = box.conf[0].cpu().numpy()

            # Class ID and name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            print(f'{class_name}: {confidence:.2f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})')

    return results


# Example usage
if __name__ == '__main__':
    results = detect_objects('test_image.jpg')
```

## YOLO ROS 2 Node

```python
#!/usr/bin/env python3
"""
YOLO Object Detection ROS 2 Node
Physical AI Book - Chapter 3

Real-time object detection for robotics applications.

Usage:
    ros2 run physical_ai_examples object_detector

    # With custom model
    ros2 run physical_ai_examples object_detector --ros-args -p model:=yolov8s.pt

Expected Output:
    [INFO] [object_detector]: Loaded model: yolov8n.pt
    [INFO] [object_detector]: Detection: cup (0.95) at [100, 200, 150, 280]
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector(Node):
    """
    YOLO-based object detection node.

    Subscribes:
        /camera/image_raw (Image): Input camera images

    Publishes:
        /detections (Detection2DArray): Detected objects
        /detection_image (Image): Annotated debug image

    Parameters:
        model (str): YOLO model file (default: yolov8n.pt)
        confidence_threshold (float): Minimum confidence (default: 0.5)
        device (str): Inference device (default: 'cpu')
    """

    def __init__(self):
        super().__init__('object_detector')

        if not YOLO_AVAILABLE:
            self.get_logger().error('ultralytics not installed!')
            self.get_logger().error('Install with: pip install ultralytics')
            return

        # Parameters
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')  # 'cpu' or 'cuda:0'

        model_path = self.get_parameter('model').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value

        # Load YOLO model
        self.get_logger().info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.get_logger().info(f'Model loaded on {device}')

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10
        )
        self.image_pub = self.create_publisher(
            Image, '/detection_image', 10
        )

        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0

        self.get_logger().info('Object detector ready')

    def image_callback(self, msg):
        """Process incoming image and detect objects."""
        import time
        start = time.time()

        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLO inference
        results = self.model(cv_image, verbose=False)

        # Process results
        detections = Detection2DArray()
        detections.header = msg.header

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box data
                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Create Detection2D message
                detection = Detection2D()
                detection.header = msg.header

                # Bounding box (center + size format)
                detection.bbox.center.position.x = float((x1 + x2) / 2)
                detection.bbox.center.position.y = float((y1 + y2) / 2)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                # Hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence
                detection.results.append(hypothesis)

                detections.detections.append(detection)

                # Log detection
                self.get_logger().debug(
                    f'Detection: {class_name} ({confidence:.2f}) at '
                    f'[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]'
                )

        # Publish detections
        self.detection_pub.publish(detections)

        # Publish annotated image
        annotated = results[0].plot() if results else cv_image
        img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)

        # Performance tracking
        elapsed = time.time() - start
        self.frame_count += 1
        self.total_time += elapsed
        if self.frame_count % 30 == 0:
            avg_fps = self.frame_count / self.total_time
            self.get_logger().info(f'Average FPS: {avg_fps:.1f}')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Detection Message Processing

```python
#!/usr/bin/env python3
"""
Detection Processor
Physical AI Book - Chapter 3

Processes detection messages for robot decision-making.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


class DetectionProcessor(Node):
    """Process detections for robotics tasks."""

    def __init__(self):
        super().__init__('detection_processor')

        # Target objects to look for
        self.declare_parameter('target_objects', ['cup', 'bottle', 'book'])
        self.targets = self.get_parameter('target_objects').value

        self.create_subscription(
            Detection2DArray, '/detections',
            self.detection_callback, 10
        )

        self.get_logger().info(f'Looking for: {self.targets}')

    def detection_callback(self, msg):
        """Process incoming detections."""
        for detection in msg.detections:
            if not detection.results:
                continue

            # Get best hypothesis
            best = max(detection.results, key=lambda h: h.hypothesis.score)
            class_name = best.hypothesis.class_id
            confidence = best.hypothesis.score

            # Check if it's a target
            if class_name in self.targets:
                cx = detection.bbox.center.position.x
                cy = detection.bbox.center.position.y
                w = detection.bbox.size_x
                h = detection.bbox.size_y

                self.get_logger().info(
                    f'Found target: {class_name} ({confidence:.2f}) '
                    f'at center ({cx:.0f}, {cy:.0f}), size {w:.0f}x{h:.0f}'
                )

                # Calculate relative position for robot
                # Assuming 640x480 image, center is (320, 240)
                dx = cx - 320  # Positive = right of center
                dy = cy - 240  # Positive = below center

                self.get_logger().info(
                    f'Offset from center: dx={dx:.0f}, dy={dy:.0f}'
                )

    def find_nearest_target(self, detections, target_class=None):
        """
        Find the detection nearest to image center.

        Args:
            detections: Detection2DArray
            target_class: Optional filter by class

        Returns:
            Nearest detection or None
        """
        image_center = (320, 240)  # Assume 640x480
        nearest = None
        min_dist = float('inf')

        for det in detections.detections:
            if not det.results:
                continue

            best = max(det.results, key=lambda h: h.hypothesis.score)

            if target_class and best.hypothesis.class_id != target_class:
                continue

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            dist = ((cx - image_center[0])**2 + (cy - image_center[1])**2)**0.5

            if dist < min_dist:
                min_dist = dist
                nearest = det

        return nearest


def main(args=None):
    rclpy.init(args=args)
    node = DetectionProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Custom Object Detection

### Training YOLO on Custom Data

```python
"""
Custom YOLO Training
Physical AI Book - Chapter 3

Train YOLO on custom robot objects.
"""

from ultralytics import YOLO


def train_custom_model():
    """Train YOLO on custom dataset."""
    # Load pretrained model
    model = YOLO('yolov8n.pt')

    # Train on custom data
    # Requires data.yaml with paths to images/labels
    results = model.train(
        data='custom_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='humanoid_objects'
    )

    return results


# Example data.yaml structure:
"""
# custom_data.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: cup
  1: bottle
  2: ball
  3: tool
"""
```

### Data Annotation Tips

For robotics applications:

1. **Variety**: Capture objects from multiple angles
2. **Lighting**: Include different lighting conditions
3. **Backgrounds**: Vary backgrounds as in deployment
4. **Occlusion**: Include partially hidden objects
5. **Scale**: Capture at various distances

## Performance Optimization

### GPU Acceleration

```python
# Check CUDA availability
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# Load model on GPU
model = YOLO('yolov8n.pt')
model.to('cuda:0')  # Use first GPU
```

### Inference Optimization

```python
def optimize_inference(model):
    """Optimize model for faster inference."""
    # Export to TensorRT (NVIDIA GPUs)
    model.export(format='engine', device=0)

    # Or export to ONNX (cross-platform)
    model.export(format='onnx')
```

### Batch Processing

```python
def batch_detect(images, model, batch_size=4):
    """
    Process multiple images in batches.

    Args:
        images: List of image paths or arrays
        model: YOLO model
        batch_size: Number of images per batch

    Returns:
        All results
    """
    all_results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        results = model(batch)
        all_results.extend(results)

    return all_results
```

## Summary

Key takeaways from this lesson:

1. **YOLO** provides real-time object detection
2. **vision_msgs** standardizes detection messages in ROS 2
3. **Confidence thresholds** filter unreliable detections
4. **GPU acceleration** dramatically improves performance
5. **Custom training** enables domain-specific detection

## Next Steps

In the [next lesson](./lesson-03-isaac-perception.md), we will:
- Explore Isaac ROS perception pipelines
- Implement depth-based perception
- Integrate with the humanoid robot

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [vision_msgs Package](https://github.com/ros-perception/vision_msgs)
- [COCO Dataset Classes](https://cocodataset.org/)
