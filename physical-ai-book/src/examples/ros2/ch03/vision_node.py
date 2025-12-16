#!/usr/bin/env python3
"""
Vision Processing Node
Physical AI Book - Chapter 3: Perception

Complete vision processing pipeline with color segmentation,
edge detection, and contour analysis.

Usage:
    ros2 run physical_ai_examples vision_node

    # With custom target color
    ros2 run physical_ai_examples vision_node --ros-args -p target_color:=blue

Expected Output:
    [INFO] [vision_node]: Vision processor ready, tracking red
    [INFO] [vision_node]: Detected 2 objects
    [INFO] [vision_node]: Object 1: area=1523, center=(234, 156)

Dependencies:
    - rclpy
    - sensor_msgs
    - cv_bridge
    - opencv-python
    - numpy

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
from typing import List, Dict, Tuple, Optional


class VisionNode(Node):
    """
    Complete vision processing pipeline.

    Subscribes:
        /camera/image_raw (Image): Input camera images

    Publishes:
        /vision/detections (String): JSON-encoded detection results
        /vision/debug_image (Image): Annotated debug image
        /vision/edges (Image): Edge detection output
        /vision/mask (Image): Color segmentation mask

    Parameters:
        target_color (str): Color to detect (red, green, blue, yellow)
        min_area (int): Minimum contour area to consider
        enable_edges (bool): Enable edge detection output
    """

    # HSV color ranges for segmentation
    COLOR_RANGES = {
        'red': {
            'lower1': np.array([0, 100, 100]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([160, 100, 100]),
            'upper2': np.array([180, 255, 255]),
        },
        'green': {
            'lower': np.array([35, 100, 100]),
            'upper': np.array([85, 255, 255]),
        },
        'blue': {
            'lower': np.array([100, 100, 100]),
            'upper': np.array([130, 255, 255]),
        },
        'yellow': {
            'lower': np.array([20, 100, 100]),
            'upper': np.array([35, 255, 255]),
        },
    }

    def __init__(self):
        super().__init__('vision_node')

        # Parameters
        self.declare_parameter('target_color', 'red')
        self.declare_parameter('min_area', 500)
        self.declare_parameter('enable_edges', True)
        self.declare_parameter('edge_low_threshold', 50)
        self.declare_parameter('edge_high_threshold', 150)

        self.target_color = self.get_parameter('target_color').value
        self.min_area = self.get_parameter('min_area').value
        self.enable_edges = self.get_parameter('enable_edges').value
        self.edge_low = self.get_parameter('edge_low_threshold').value
        self.edge_high = self.get_parameter('edge_high_threshold').value

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(String, '/vision/detections', 10)
        self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)
        self.edge_pub = self.create_publisher(Image, '/vision/edges', 10)
        self.mask_pub = self.create_publisher(Image, '/vision/mask', 10)

        # Statistics
        self.frame_count = 0

        self.get_logger().info(f'Vision processor ready, tracking {self.target_color}')

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        self.frame_count += 1

        # Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Create debug image
        debug_image = cv_image.copy()

        # 1. Color segmentation
        mask = self.segment_color(cv_image, self.target_color)

        # 2. Morphological operations to clean mask
        mask = self.clean_mask(mask)

        # 3. Find and analyze contours
        detections = self.find_objects(mask, cv_image.shape)

        # 4. Draw detections on debug image
        debug_image = self.draw_detections(debug_image, detections)

        # 5. Edge detection (optional)
        if self.enable_edges:
            edges = self.detect_edges(cv_image)
            edge_msg = self.bridge.cv2_to_imgmsg(edges, 'mono8')
            edge_msg.header = msg.header
            self.edge_pub.publish(edge_msg)

        # Publish results
        self.publish_results(msg.header, detections, debug_image, mask)

        # Log periodically
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Processed {self.frame_count} frames')

    def segment_color(self, image: np.ndarray, color: str) -> np.ndarray:
        """
        Segment image by color using HSV color space.

        Args:
            image: BGR image
            color: Target color name

        Returns:
            Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color not in self.COLOR_RANGES:
            self.get_logger().warn(f'Unknown color: {color}')
            return np.zeros(image.shape[:2], dtype=np.uint8)

        ranges = self.COLOR_RANGES[color]

        if color == 'red':
            # Red wraps around in HSV
            mask1 = cv2.inRange(hsv, ranges['lower1'], ranges['upper1'])
            mask2 = cv2.inRange(hsv, ranges['lower2'], ranges['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])

        return mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up mask.

        Args:
            mask: Binary mask

        Returns:
            Cleaned mask
        """
        kernel = np.ones((5, 5), np.uint8)

        # Opening removes small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Closing fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_objects(
        self,
        mask: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> List[Dict]:
        """
        Find objects in mask and extract properties.

        Args:
            mask: Binary mask
            image_shape: Original image shape for normalization

        Returns:
            List of detection dictionaries
        """
        detections = []

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = image_shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Additional properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = float(w) / h if h > 0 else 0

            # Normalized coordinates (0-1)
            cx_norm = cx / width
            cy_norm = cy / height

            detection = {
                'color': self.target_color,
                'center': [cx, cy],
                'center_normalized': [cx_norm, cy_norm],
                'bbox': [x, y, w, h],
                'area': int(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
            }

            detections.append(detection)

        return detections

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny algorithm.

        Args:
            image: BGR image

        Returns:
            Binary edge image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
        return edges

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            image: BGR image to annotate
            detections: List of detections

        Returns:
            Annotated image
        """
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            cx, cy = det['center']

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            # Draw label
            label = f"{det['color']} #{i+1}"
            cv2.putText(
                image, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            # Draw area
            area_label = f"A={det['area']}"
            cv2.putText(
                image, area_label,
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
            )

        # Draw detection count
        cv2.putText(
            image, f"Objects: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )

        return image

    def publish_results(
        self,
        header,
        detections: List[Dict],
        debug_image: np.ndarray,
        mask: np.ndarray
    ):
        """Publish all results."""
        # Detection JSON
        detection_msg = String()
        detection_msg.data = json.dumps({
            'frame': self.frame_count,
            'target_color': self.target_color,
            'count': len(detections),
            'detections': detections,
        })
        self.detection_pub.publish(detection_msg)

        # Debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)

        # Mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, 'mono8')
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

        # Log detections
        if detections:
            self.get_logger().debug(f'Detected {len(detections)} objects')


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = VisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
