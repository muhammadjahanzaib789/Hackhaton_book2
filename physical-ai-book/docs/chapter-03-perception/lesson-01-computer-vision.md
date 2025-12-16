---
sidebar_position: 1
title: "Lesson 1: Computer Vision Fundamentals"
description: "Image processing basics with OpenCV for robotics"
---

# Computer Vision Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

1. Process camera images using OpenCV in ROS 2
2. Apply common image transformations and filters
3. Detect features, edges, and contours
4. Build a real-time vision processing pipeline

## Prerequisites

- Completed Chapter 2 (Simulation)
- Python basics with NumPy
- Camera publishing data in Gazebo

## Introduction

Computer vision enables robots to understand their environment through cameras. For humanoid robots, vision is essential for:

- **Object Recognition**: Finding items to manipulate
- **Navigation**: Avoiding obstacles, finding paths
- **Human Interaction**: Recognizing gestures, faces
- **Task Verification**: Confirming actions completed

```
┌─────────────────────────────────────────────────────────────┐
│              Vision Pipeline for Robotics                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Camera ──▶ Preprocessing ──▶ Feature ──▶ Understanding    │
│   Input       (filters)      Extraction    (semantics)      │
│                                                             │
│  ┌───────┐   ┌───────────┐   ┌─────────┐   ┌───────────┐  │
│  │ RGB   │──▶│ Denoise   │──▶│ Edges   │──▶│ Objects   │  │
│  │ Image │   │ Resize    │   │ Corners │   │ Poses     │  │
│  │       │   │ Color cvt │   │ Blobs   │   │ Classes   │  │
│  └───────┘   └───────────┘   └─────────┘   └───────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## OpenCV in ROS 2

### Installing OpenCV

```bash
# OpenCV is included in ROS 2, but ensure cv_bridge is available
sudo apt install ros-humble-cv-bridge python3-opencv

# Verify installation
python3 -c "import cv2; print(cv2.__version__)"
```

### Converting ROS Images to OpenCV

```python
#!/usr/bin/env python3
"""
ROS 2 to OpenCV Image Conversion
Physical AI Book - Chapter 3
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageConverter(Node):
    """Converts ROS images to OpenCV format."""

    def __init__(self):
        super().__init__('image_converter')

        # CV Bridge for conversion
        self.bridge = CvBridge()

        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Image converter ready')

    def image_callback(self, msg):
        """Convert and process incoming image."""
        try:
            # Convert ROS Image to OpenCV (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Now cv_image is a NumPy array
            height, width, channels = cv_image.shape
            self.get_logger().info(f'Image: {width}x{height}x{channels}')

            # Process the image...
            processed = self.process_image(cv_image)

            # Display (for debugging)
            cv2.imshow('Camera', processed)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Conversion failed: {e}')

    def process_image(self, image):
        """Apply processing to image."""
        # Placeholder - will implement in following sections
        return image
```

## Image Preprocessing

### Color Space Conversion

Different color spaces reveal different information:

```python
def convert_color_spaces(image):
    """
    Convert image to various color spaces.

    Args:
        image: BGR image from OpenCV

    Returns:
        dict: Images in different color spaces
    """
    return {
        'bgr': image,
        'rgb': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        'gray': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        'hsv': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        'lab': cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
    }
```

| Color Space | Use Case | Components |
|-------------|----------|------------|
| **BGR/RGB** | Display, deep learning | Blue, Green, Red |
| **Grayscale** | Edge detection, features | Intensity |
| **HSV** | Color segmentation | Hue, Saturation, Value |
| **LAB** | Color difference | Lightness, A, B |

### Color-Based Segmentation

```python
def segment_by_color(image, color='red'):
    """
    Segment objects by color using HSV.

    Args:
        image: BGR image
        color: 'red', 'green', 'blue', 'yellow'

    Returns:
        Binary mask of matching pixels
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color ranges in HSV
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),      # Lower red
        'red2': ([160, 100, 100], [180, 255, 255]),  # Upper red
        'green': ([35, 100, 100], [85, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'yellow': ([20, 100, 100], [35, 255, 255]),
    }

    if color == 'red':
        # Red wraps around in HSV, need two ranges
        lower1, upper1 = color_ranges['red']
        lower2, upper2 = color_ranges['red2']
        mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    return mask
```

### Noise Reduction

```python
def reduce_noise(image, method='gaussian'):
    """
    Apply noise reduction filters.

    Args:
        image: Input image
        method: 'gaussian', 'median', 'bilateral'

    Returns:
        Filtered image
    """
    if method == 'gaussian':
        # Good general-purpose smoothing
        return cv2.GaussianBlur(image, (5, 5), 0)

    elif method == 'median':
        # Good for salt-and-pepper noise
        return cv2.medianBlur(image, 5)

    elif method == 'bilateral':
        # Preserves edges while smoothing
        return cv2.bilateralFilter(image, 9, 75, 75)

    return image
```

## Edge Detection

Edges are fundamental features for understanding scene structure.

### Canny Edge Detector

```python
def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """
    Detect edges using Canny algorithm.

    Args:
        image: Grayscale image
        low_threshold: Lower hysteresis threshold
        high_threshold: Upper hysteresis threshold

    Returns:
        Binary edge image
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges
```

### Sobel Gradients

```python
def compute_gradients(image):
    """
    Compute image gradients using Sobel operators.

    Args:
        image: Grayscale image

    Returns:
        tuple: (gradient_x, gradient_y, magnitude, direction)
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    return grad_x, grad_y, magnitude, direction
```

## Contour Detection

Contours are boundaries of connected regions - useful for finding objects.

```python
def find_contours(image, min_area=100):
    """
    Find contours in a binary or grayscale image.

    Args:
        image: Binary or grayscale image
        min_area: Minimum contour area to keep

    Returns:
        list: Filtered contours
    """
    # Ensure binary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold if not binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,      # Only external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress segments
    )

    # Filter by area
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

    return filtered


def analyze_contour(contour):
    """
    Extract properties from a contour.

    Args:
        contour: OpenCV contour

    Returns:
        dict: Contour properties
    """
    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(contour, closed=True)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Circularity (1.0 = perfect circle)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # Aspect ratio
    aspect_ratio = float(w) / h if h > 0 else 0

    return {
        'area': area,
        'perimeter': perimeter,
        'bounding_box': (x, y, w, h),
        'centroid': (cx, cy),
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
    }
```

## Feature Detection

Features are distinctive points used for recognition and tracking.

### Corner Detection (Harris)

```python
def detect_corners_harris(image, block_size=2, ksize=3, k=0.04):
    """
    Detect corners using Harris corner detector.

    Args:
        image: Grayscale image
        block_size: Neighborhood size
        ksize: Sobel kernel size
        k: Harris detector free parameter

    Returns:
        Corner response image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = np.float32(gray)

    # Harris corner detection
    corners = cv2.cornerHarris(gray, block_size, ksize, k)

    # Dilate to mark corners
    corners = cv2.dilate(corners, None)

    return corners
```

### ORB Features

```python
def detect_orb_features(image, n_features=500):
    """
    Detect ORB features (fast and rotation-invariant).

    Args:
        image: Input image
        n_features: Maximum number of features

    Returns:
        tuple: (keypoints, descriptors)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)

    # Detect and compute
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(desc1, desc2, ratio_threshold=0.75):
    """
    Match ORB features between two images.

    Args:
        desc1: Descriptors from image 1
        desc2: Descriptors from image 2
        ratio_threshold: Lowe's ratio test threshold

    Returns:
        list: Good matches
    """
    # Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Find matches
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches
```

## Complete Vision Node

```python
#!/usr/bin/env python3
"""
Complete Vision Processing Node
Physical AI Book - Chapter 3

Demonstrates a full vision pipeline with multiple processing stages.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json


class VisionProcessor(Node):
    """Complete vision processing pipeline."""

    def __init__(self):
        super().__init__('vision_processor')

        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('target_color', 'red')
        self.declare_parameter('min_area', 500)
        self.target_color = self.get_parameter('target_color').value
        self.min_area = self.get_parameter('min_area').value

        # Subscribers
        self.create_subscription(
            Image, '/camera/image_raw',
            self.image_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            String, '/vision/detections', 10
        )
        self.debug_pub = self.create_publisher(
            Image, '/vision/debug_image', 10
        )

        self.get_logger().info(f'Vision processor ready, tracking {self.target_color}')

    def image_callback(self, msg):
        """Process incoming image."""
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run detection pipeline
        detections, debug_image = self.detect_objects(cv_image)

        # Publish results
        if detections:
            detection_msg = String()
            detection_msg.data = json.dumps(detections)
            self.detection_pub.publish(detection_msg)

        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
        self.debug_pub.publish(debug_msg)

    def detect_objects(self, image):
        """
        Run full detection pipeline.

        Returns:
            tuple: (detections list, debug image)
        """
        detections = []
        debug_image = image.copy()

        # 1. Color segmentation
        mask = self.segment_by_color(image, self.target_color)

        # 2. Noise reduction
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

        # 3. Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 4. Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Get properties
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else x + w // 2
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else y + h // 2

            detection = {
                'color': self.target_color,
                'center': [cx, cy],
                'bbox': [x, y, w, h],
                'area': area,
            }
            detections.append(detection)

            # Draw on debug image
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_image, f'{self.target_color}',
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

        return detections, debug_image

    def segment_by_color(self, image, color):
        """Segment image by color."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255]),
        }

        if color == 'red':
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            return cv2.bitwise_or(mask1, mask2)
        elif color in color_ranges:
            lower, upper = color_ranges[color]
            return cv2.inRange(hsv, np.array(lower), np.array(upper))
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)


def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **cv_bridge** converts between ROS Image and OpenCV formats
2. **Color spaces** (HSV) enable robust color-based segmentation
3. **Edge detection** reveals scene structure
4. **Contours** identify object boundaries
5. **Features** (ORB) enable recognition and tracking

## Next Steps

In the [next lesson](./lesson-02-object-detection.md), we will:
- Implement deep learning-based object detection
- Use YOLO for real-time detection
- Integrate detection with ROS 2

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [cv_bridge Tutorial](https://wiki.ros.org/cv_bridge/Tutorials)
- [Image Processing Algorithms](https://homepages.inf.ed.ac.uk/rbf/HIPR2/)
