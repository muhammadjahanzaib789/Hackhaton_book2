#!/usr/bin/env python3
"""
YOLO Object Detector Node
Physical AI Book - Chapter 3: Perception

Real-time object detection using YOLO with ROS 2 integration.
Publishes standard vision_msgs for downstream processing.

Usage:
    ros2 run physical_ai_examples object_detector

    # With custom model and confidence
    ros2 run physical_ai_examples object_detector --ros-args \
        -p model:=yolov8s.pt -p confidence:=0.6

Expected Output:
    [INFO] [object_detector]: Loading model: yolov8n.pt
    [INFO] [object_detector]: Object detector ready
    [INFO] [object_detector]: Detected: cup (0.92) at [123, 456]

Dependencies:
    - rclpy
    - sensor_msgs
    - vision_msgs
    - cv_bridge
    - ultralytics (YOLO)

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from typing import List, Dict, Optional

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try importing vision_msgs
try:
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False


class ObjectDetectorNode(Node):
    """
    YOLO-based object detection node for robotics.

    Subscribes:
        /camera/image_raw (Image): Input camera images

    Publishes:
        /detections (Detection2DArray or String): Detected objects
        /detection_image (Image): Annotated image

    Parameters:
        model (str): YOLO model name (default: yolov8n.pt)
        confidence (float): Minimum confidence threshold (default: 0.5)
        device (str): Inference device - 'cpu' or 'cuda:0' (default: cpu)
        classes (list): Filter specific classes (default: all)
        publish_rate (float): Max publish rate in Hz (default: 30.0)
    """

    def __init__(self):
        super().__init__('object_detector')

        # Check dependencies
        if not YOLO_AVAILABLE:
            self.get_logger().error(
                'ultralytics not installed! '
                'Install with: pip install ultralytics'
            )
            return

        # Parameters
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('classes', [])
        self.declare_parameter('publish_rate', 30.0)

        model_name = self.get_parameter('model').value
        self.confidence = self.get_parameter('confidence').value
        device = self.get_parameter('device').value
        self.filter_classes = self.get_parameter('classes').value
        self.min_period = 1.0 / self.get_parameter('publish_rate').value

        # Load model
        self.get_logger().info(f'Loading model: {model_name}')
        try:
            self.model = YOLO(model_name)
            self.model.to(device)
            self.class_names = self.model.names
            self.get_logger().info(f'Model loaded on {device}')
            self.get_logger().info(f'Classes: {len(self.class_names)} available')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return

        # CV Bridge
        self.bridge = CvBridge()

        # Timing
        self.last_process_time = 0.0

        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0

        # Subscribers
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        # Publishers
        if VISION_MSGS_AVAILABLE:
            self.detection_pub = self.create_publisher(
                Detection2DArray, '/detections', 10
            )
        else:
            # Fallback to JSON string
            self.detection_pub = self.create_publisher(
                String, '/detections', 10
            )
            self.get_logger().warn(
                'vision_msgs not available, using JSON format'
            )

        self.image_pub = self.create_publisher(
            Image, '/detection_image', 10
        )

        self.get_logger().info('Object detector ready')

    def image_callback(self, msg: Image):
        """Process incoming image for object detection."""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_process_time < self.min_period:
            return
        self.last_process_time = current_time

        # Convert to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Run inference
        start_time = time.time()
        results = self.model(
            cv_image,
            conf=self.confidence,
            verbose=False
        )
        inference_time = time.time() - start_time

        # Update statistics
        self.frame_count += 1
        self.total_inference_time += inference_time

        # Process results
        detections = self.process_results(results, msg.header)

        # Publish detections
        self.publish_detections(detections, msg.header)

        # Publish annotated image
        annotated = results[0].plot() if results else cv_image
        img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)

        # Log statistics periodically
        if self.frame_count % 30 == 0:
            avg_time = self.total_inference_time / self.frame_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{len(detections)} objects, '
                f'{fps:.1f} FPS, '
                f'{inference_time*1000:.1f}ms'
            )

    def process_results(self, results, header) -> List[Dict]:
        """
        Process YOLO results into detection list.

        Args:
            results: YOLO results object
            header: ROS message header

        Returns:
            List of detection dictionaries
        """
        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get detection data
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]

                # Filter by class if specified
                if self.filter_classes and class_name not in self.filter_classes:
                    continue

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'center_x': float((x1 + x2) / 2),
                        'center_y': float((y1 + y2) / 2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                    }
                }

                detections.append(detection)

                # Log high-confidence detections
                if confidence > 0.7:
                    self.get_logger().debug(
                        f'Detected: {class_name} ({confidence:.2f}) '
                        f'at [{x1:.0f}, {y1:.0f}]'
                    )

        return detections

    def publish_detections(self, detections: List[Dict], header):
        """Publish detections in appropriate format."""
        if VISION_MSGS_AVAILABLE:
            self.publish_vision_msgs(detections, header)
        else:
            self.publish_json(detections, header)

    def publish_vision_msgs(self, detections: List[Dict], header):
        """Publish using vision_msgs format."""
        msg = Detection2DArray()
        msg.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box
            bbox = det['bbox']
            detection.bbox.center.position.x = bbox['center_x']
            detection.bbox.center.position.y = bbox['center_y']
            detection.bbox.size_x = bbox['width']
            detection.bbox.size_y = bbox['height']

            # Hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            msg.detections.append(detection)

        self.detection_pub.publish(msg)

    def publish_json(self, detections: List[Dict], header):
        """Publish using JSON format (fallback)."""
        msg = String()
        msg.data = json.dumps({
            'timestamp': header.stamp.sec + header.stamp.nanosec * 1e-9,
            'frame_id': header.frame_id,
            'count': len(detections),
            'detections': detections,
        })
        self.detection_pub.publish(msg)

    def get_class_names(self) -> List[str]:
        """Get list of all class names."""
        return list(self.class_names.values())


class MockObjectDetector(Node):
    """
    Mock object detector for testing without YOLO.
    Generates synthetic detections.
    """

    def __init__(self):
        super().__init__('mock_object_detector')

        self.bridge = CvBridge()
        self.frame_count = 0

        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        self.detection_pub = self.create_publisher(
            String, '/detections', 10
        )

        self.get_logger().info('Mock object detector ready (no YOLO)')

    def image_callback(self, msg: Image):
        """Generate mock detections."""
        self.frame_count += 1

        # Generate synthetic detection
        import random
        if random.random() > 0.3:  # 70% chance of detection
            detections = [{
                'class_name': random.choice(['cup', 'bottle', 'book']),
                'confidence': random.uniform(0.7, 0.99),
                'bbox': {
                    'center_x': random.randint(100, 540),
                    'center_y': random.randint(100, 380),
                    'width': random.randint(50, 150),
                    'height': random.randint(50, 150),
                }
            }]
        else:
            detections = []

        # Publish
        msg = String()
        msg.data = json.dumps({
            'frame': self.frame_count,
            'count': len(detections),
            'detections': detections,
        })
        self.detection_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    if YOLO_AVAILABLE:
        node = ObjectDetectorNode()
    else:
        print('YOLO not available, using mock detector')
        node = MockObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
