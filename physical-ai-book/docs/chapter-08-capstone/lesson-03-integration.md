---
sidebar_position: 3
title: "Lesson 3: System Integration"
description: "Integrating all subsystems into a complete Physical AI Assistant"
---

# System Integration

## Overview

In this lesson, we integrate all robot subsystems into a cohesive system:

1. **Perception Integration** - Object detection and scene understanding
2. **Navigation Integration** - Nav2 with coordinator
3. **Manipulation Integration** - MoveIt2 arm control
4. **LLM Integration** - Natural language planning

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Integrated System Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    COORDINATOR                       │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐        │   │
│  │  │   State   │ │   Task    │ │  Safety   │        │   │
│  │  │  Machine  │ │  Queue    │ │  Monitor  │        │   │
│  │  └───────────┘ └───────────┘ └───────────┘        │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│         ┌───────────────┼───────────────┐                  │
│         │               │               │                   │
│         ▼               ▼               ▼                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │   Voice    │  │    LLM     │  │ Perception │           │
│  │ Interface  │  │  Planner   │  │  Pipeline  │           │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│        │               │               │                    │
│        │               │               │                    │
│  ┌─────┴───────────────┴───────────────┴─────┐            │
│  │              ACTION LAYER                  │            │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │            │
│  │  │  Nav2    │ │ MoveIt2  │ │ Gripper  │  │            │
│  │  │  Client  │ │  Client  │ │  Client  │  │            │
│  │  └──────────┘ └──────────┘ └──────────┘  │            │
│  └───────────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Perception Integration

```python
#!/usr/bin/env python3
"""
Perception Integration
Physical AI Book - Chapter 8: Capstone

Integrates object detection with coordinator for scene understanding.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from typing import Dict, List, Optional
import json


class PerceptionNode(Node):
    """
    Perception node for object detection and tracking.

    Publishes detected objects for coordinator use.
    """

    def __init__(self):
        super().__init__('perception')

        # CV Bridge
        self.bridge = CvBridge()

        # Object tracker state
        self.tracked_objects: Dict[str, Dict] = {}
        self.target_object: Optional[str] = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self._image_callback, 10
        )
        self.target_sub = self.create_subscription(
            String, '/perception/set_target',
            self._set_target_callback, 10
        )

        # Publishers
        self.detections_pub = self.create_publisher(
            Detection2DArray, '/perception/detections', 10
        )
        self.objects_pub = self.create_publisher(
            String, '/perception/objects', 10
        )
        self.target_pose_pub = self.create_publisher(
            PointStamped, '/perception/target_pose', 10
        )

        # Detection model (would be real model in production)
        self.detector = SimpleDetector()

        self.get_logger().info('Perception node ready')

    def _image_callback(self, msg: Image):
        """Process incoming images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            detections = self.detector.detect(cv_image)

            # Update tracked objects
            self._update_tracking(detections)

            # Publish objects
            self._publish_objects()

            # Check for target
            if self.target_object:
                self._publish_target_pose()

        except Exception as e:
            self.get_logger().error(f'Perception error: {e}')

    def _update_tracking(self, detections: List[Dict]):
        """Update object tracking."""
        for det in detections:
            obj_id = det['class']
            self.tracked_objects[obj_id] = {
                'class': det['class'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'position': det.get('position', [0, 0, 0]),
                'last_seen': self.get_clock().now()
            }

    def _publish_objects(self):
        """Publish detected objects."""
        msg = String()
        msg.data = json.dumps(list(self.tracked_objects.values()), default=str)
        self.objects_pub.publish(msg)

    def _set_target_callback(self, msg: String):
        """Set target object to track."""
        self.target_object = msg.data
        self.get_logger().info(f'Target set: {self.target_object}')

    def _publish_target_pose(self):
        """Publish target object pose."""
        if self.target_object not in self.tracked_objects:
            return

        obj = self.tracked_objects[self.target_object]
        pos = obj.get('position', [0, 0, 0])

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])

        self.target_pose_pub.publish(msg)


class SimpleDetector:
    """Simple detector for demo (replace with real model)."""

    CLASSES = ['cup', 'bottle', 'ball', 'box', 'person']

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image."""
        # Placeholder - would use real detection model
        return [
            {
                'class': 'cup',
                'confidence': 0.95,
                'bbox': [100, 100, 50, 50],
                'position': [0.5, 0.0, 0.3]
            }
        ]
```

## Navigation Integration

```python
"""
Navigation Integration
Physical AI Book - Chapter 8: Capstone

Wrapper for Nav2 integration with coordinator.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from typing import Optional, Callable
import json


class NavigationManager:
    """
    Navigation manager for coordinator integration.

    Provides high-level navigation interface.
    """

    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()

        # Action clients
        self.nav_client = ActionClient(
            node, NavigateToPose, 'navigate_to_pose'
        )
        self.waypoint_client = ActionClient(
            node, NavigateThroughPoses, 'navigate_through_poses'
        )

        # State
        self.is_navigating = False
        self.current_goal = None

        # Callbacks
        self.on_complete: Optional[Callable] = None
        self.on_feedback: Optional[Callable] = None

    async def navigate_to(
        self,
        x: float, y: float, yaw: float = 0.0,
        frame_id: str = 'map'
    ) -> bool:
        """
        Navigate to a pose.

        Args:
            x, y: Target position
            yaw: Target orientation
            frame_id: Reference frame

        Returns:
            True if navigation started successfully
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.logger.error('Navigation server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = self._create_pose(x, y, yaw, frame_id)

        self.logger.info(f'Navigating to ({x:.2f}, {y:.2f})')

        self.is_navigating = True
        self.current_goal = goal

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

        return True

    async def navigate_through(
        self,
        waypoints: list,
        frame_id: str = 'map'
    ) -> bool:
        """Navigate through multiple waypoints."""
        if not self.waypoint_client.wait_for_server(timeout_sec=5.0):
            return False

        goal = NavigateThroughPoses.Goal()
        for x, y, yaw in waypoints:
            goal.poses.append(self._create_pose(x, y, yaw, frame_id))

        self.is_navigating = True

        future = self.waypoint_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

        return True

    def cancel(self):
        """Cancel current navigation."""
        if self.is_navigating and self.current_goal:
            self.logger.info('Canceling navigation')
            # Would cancel the goal handle

    def _create_pose(
        self, x: float, y: float, yaw: float, frame_id: str
    ) -> PoseStamped:
        """Create PoseStamped message."""
        import math

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        return pose

    def _feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        if self.on_feedback:
            self.on_feedback(feedback_msg.feedback)

    def _goal_response_callback(self, future):
        """Handle goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.is_navigating = False
            if self.on_complete:
                self.on_complete(False, "Goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """Handle navigation result."""
        self.is_navigating = False
        result = future.result()

        success = result.status == 4  # SUCCEEDED
        if self.on_complete:
            self.on_complete(success, "" if success else "Navigation failed")
```

## Manipulation Integration

```python
"""
Manipulation Integration
Physical AI Book - Chapter 8: Capstone

Wrapper for MoveIt2 integration with coordinator.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64MultiArray
from typing import Optional, List


class ManipulationManager:
    """
    Manipulation manager for arm control.

    Provides high-level manipulation interface.
    """

    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()

        # Gripper publisher
        self.gripper_pub = node.create_publisher(
            Float64MultiArray, '/gripper_controller/command', 10
        )

        # State
        self.is_moving = False
        self.gripper_state = 0.0  # 0=closed, 1=open

    async def move_to_pose(
        self,
        position: List[float],
        orientation: Optional[List[float]] = None
    ) -> bool:
        """
        Move end effector to target pose.

        Args:
            position: [x, y, z] target position
            orientation: [x, y, z, w] quaternion (optional)

        Returns:
            True if motion started
        """
        self.logger.info(f'Moving to position: {position}')

        # In production, would use MoveIt2 action
        self.is_moving = True

        # Simulate motion
        await self._simulate_motion()

        self.is_moving = False
        return True

    async def pick(self, position: List[float]) -> bool:
        """
        Pick object at position.

        Args:
            position: Object position

        Returns:
            True if pick succeeded
        """
        self.logger.info(f'Picking at: {position}')

        # Approach
        approach_pos = position.copy()
        approach_pos[2] += 0.1  # 10cm above
        await self.move_to_pose(approach_pos)

        # Open gripper
        self.open_gripper()

        # Descend
        await self.move_to_pose(position)

        # Close gripper
        self.close_gripper()

        # Lift
        await self.move_to_pose(approach_pos)

        return True

    async def place(self, position: List[float]) -> bool:
        """
        Place object at position.

        Args:
            position: Target position

        Returns:
            True if place succeeded
        """
        self.logger.info(f'Placing at: {position}')

        # Approach
        approach_pos = position.copy()
        approach_pos[2] += 0.1
        await self.move_to_pose(approach_pos)

        # Descend
        await self.move_to_pose(position)

        # Open gripper
        self.open_gripper()

        # Retreat
        await self.move_to_pose(approach_pos)

        return True

    def open_gripper(self):
        """Open the gripper."""
        msg = Float64MultiArray()
        msg.data = [1.0]  # Open
        self.gripper_pub.publish(msg)
        self.gripper_state = 1.0
        self.logger.debug('Gripper opened')

    def close_gripper(self):
        """Close the gripper."""
        msg = Float64MultiArray()
        msg.data = [0.0]  # Closed
        self.gripper_pub.publish(msg)
        self.gripper_state = 0.0
        self.logger.debug('Gripper closed')

    async def _simulate_motion(self):
        """Simulate arm motion for demo."""
        import asyncio
        await asyncio.sleep(1.0)
```

## LLM Planner Integration

```python
"""
LLM Planner Integration
Physical AI Book - Chapter 8: Capstone

Integrates LLM-based task planning with coordinator.
"""

from typing import Dict, List, Optional, Any
import json


class LLMTaskPlanner:
    """
    LLM-based task planner for natural language commands.

    Converts voice commands to executable action sequences.
    """

    # Known locations in the environment
    LOCATIONS = {
        'kitchen': {'x': 5.0, 'y': 2.0, 'yaw': 0.0},
        'living room': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        'bedroom': {'x': -3.0, 'y': 4.0, 'yaw': 1.57},
        'bathroom': {'x': -3.0, 'y': -2.0, 'yaw': 3.14},
        'home': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
    }

    # Known objects
    OBJECTS = ['cup', 'bottle', 'ball', 'remote', 'book', 'phone']

    def __init__(self, llm_provider=None):
        """
        Initialize planner.

        Args:
            llm_provider: LLM provider for planning (optional)
        """
        self.llm = llm_provider

    def plan(self, command: str, world_state: Optional[Dict] = None) -> Dict:
        """
        Generate action plan from natural language command.

        Args:
            command: Natural language command
            world_state: Current world state

        Returns:
            Plan dictionary with goal and actions
        """
        if self.llm:
            return self._llm_plan(command, world_state)
        else:
            return self._rule_based_plan(command)

    def _llm_plan(self, command: str, world_state: Dict) -> Dict:
        """Use LLM for planning."""
        prompt = self._build_prompt(command, world_state)

        # Would call LLM here
        # response = self.llm.generate(prompt)

        # For demo, use rule-based fallback
        return self._rule_based_plan(command)

    def _rule_based_plan(self, command: str) -> Dict:
        """Simple rule-based planning."""
        cmd = command.lower()

        # Navigation
        for loc_name, loc_pose in self.LOCATIONS.items():
            if loc_name in cmd and ('go' in cmd or 'navigate' in cmd or 'come' in cmd):
                return {
                    'goal': f'Navigate to {loc_name}',
                    'actions': [
                        {
                            'type': 'navigate',
                            'target': loc_pose,
                            'location_name': loc_name
                        },
                        {
                            'type': 'speak',
                            'message': f'I have arrived at the {loc_name}'
                        }
                    ]
                }

        # Fetch/bring
        if 'bring' in cmd or 'fetch' in cmd or 'get' in cmd:
            for obj in self.OBJECTS:
                if obj in cmd:
                    return self._create_fetch_plan(obj)

            return {
                'goal': 'Fetch item',
                'actions': [
                    {'type': 'speak', 'message': 'What would you like me to get?'}
                ]
            }

        # Pick up
        if 'pick' in cmd and 'up' in cmd:
            for obj in self.OBJECTS:
                if obj in cmd:
                    return {
                        'goal': f'Pick up {obj}',
                        'actions': [
                            {'type': 'find_object', 'object': obj},
                            {'type': 'navigate_to_object'},
                            {'type': 'pick', 'object': obj},
                            {'type': 'speak', 'message': f'I have picked up the {obj}'}
                        ]
                    }

        # Unknown command
        return {
            'goal': 'Unknown',
            'actions': [
                {
                    'type': 'speak',
                    'message': "I'm not sure how to do that. Can you rephrase?"
                }
            ]
        }

    def _create_fetch_plan(self, obj: str) -> Dict:
        """Create a fetch plan for an object."""
        return {
            'goal': f'Fetch {obj}',
            'actions': [
                {'type': 'speak', 'message': f'I will get the {obj} for you'},
                {'type': 'navigate', 'target': self.LOCATIONS['kitchen'],
                 'location_name': 'kitchen'},
                {'type': 'find_object', 'object': obj},
                {'type': 'pick', 'object': obj},
                {'type': 'navigate', 'target': self.LOCATIONS['home'],
                 'location_name': 'home'},
                {'type': 'speak', 'message': f'Here is the {obj}'}
            ]
        }

    def _build_prompt(self, command: str, world_state: Dict) -> str:
        """Build LLM prompt."""
        return f"""You are a robot task planner.

Available locations: {list(self.LOCATIONS.keys())}
Available objects: {self.OBJECTS}

World state: {json.dumps(world_state)}

User command: {command}

Generate an action plan as JSON."""
```

## Launch File

```python
#!/usr/bin/env python3
"""
Full System Launch
Physical AI Book - Chapter 8: Capstone

Launches all nodes for the complete system.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    return LaunchDescription([
        # Coordinator
        Node(
            package='home_assistant_robot',
            executable='coordinator',
            name='coordinator',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # Perception
        Node(
            package='home_assistant_robot',
            executable='perception_node',
            name='perception',
            output='screen'
        ),

        # Voice Interface
        Node(
            package='home_assistant_robot',
            executable='voice_interface',
            name='voice_interface',
            output='screen'
        ),

        # Safety Monitor
        Node(
            package='home_assistant_robot',
            executable='safety_monitor',
            name='safety_monitor',
            output='screen'
        ),
    ])
```

## Summary

In this lesson, we integrated:

1. **Perception** with object detection and tracking
2. **Navigation** with Nav2 action client
3. **Manipulation** with gripper control
4. **LLM Planning** for natural language understanding

## Next Steps

Continue to [Lesson 4](./lesson-04-testing.md) to:
- Write unit and integration tests
- Test in simulation
- Debug common issues
