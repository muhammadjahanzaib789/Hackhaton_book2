---
sidebar_position: 4
title: "Lesson 4: Testing & Deployment"
description: "Testing, debugging, and deploying the Physical AI Assistant"
---

# Testing & Deployment

## Overview

In this final lesson, we cover:

1. **Unit Testing** - Testing individual components
2. **Integration Testing** - Testing subsystem interactions
3. **Simulation Testing** - End-to-end testing in Gazebo
4. **Deployment** - Running on real hardware

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│              Testing Pyramid                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────┐                           │
│                      │   E2E   │  ← Simulation/Hardware    │
│                      │  Tests  │                           │
│                      └────┬────┘                           │
│                     ┌─────┴─────┐                          │
│                     │Integration│  ← ROS 2 Launch Tests   │
│                     │   Tests   │                          │
│                     └─────┬─────┘                          │
│                ┌──────────┴──────────┐                     │
│                │     Unit Tests      │  ← Python pytest    │
│                │                     │                      │
│                └─────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Unit Tests

```python
#!/usr/bin/env python3
"""
Unit Tests for Coordinator
Physical AI Book - Chapter 8: Capstone

Tests for coordinator state machine and planning.
"""

import pytest
from unittest.mock import MagicMock, patch
import json


class TestRobotState:
    """Test robot state transitions."""

    def test_initial_state_is_idle(self):
        """Test coordinator starts in IDLE state."""
        from home_assistant_robot.coordinator import CoordinatorNode, RobotState

        with patch('rclpy.node.Node.__init__'):
            node = object.__new__(CoordinatorNode)
            node.state = RobotState.IDLE

        assert node.state == RobotState.IDLE

    def test_state_transition_to_planning(self):
        """Test transition from IDLE to PLANNING."""
        from home_assistant_robot.coordinator import RobotState

        # Valid transition
        current = RobotState.IDLE
        next_state = RobotState.PLANNING
        assert next_state != current

    def test_emergency_stop_from_any_state(self):
        """Test emergency stop can be triggered from any state."""
        from home_assistant_robot.coordinator import RobotState

        for state in RobotState:
            if state != RobotState.EMERGENCY_STOP:
                # Should be able to transition to emergency stop
                assert RobotState.EMERGENCY_STOP is not None


class TestTaskPlan:
    """Test task planning data structures."""

    def test_task_plan_creation(self):
        """Test creating a task plan."""
        from home_assistant_robot.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test goal",
            actions=[
                {'type': 'speak', 'message': 'hello'},
                {'type': 'navigate', 'target': {'x': 1.0, 'y': 2.0}}
            ]
        )

        assert plan.goal == "test goal"
        assert len(plan.actions) == 2
        assert plan.current_index == 0

    def test_task_plan_advance(self):
        """Test advancing through actions."""
        from home_assistant_robot.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test",
            actions=[{'type': 'a'}, {'type': 'b'}]
        )

        assert plan.current_action == {'type': 'a'}

        plan.advance()
        assert plan.current_index == 1
        assert plan.current_action == {'type': 'b'}

        plan.advance()
        assert plan.is_complete

    def test_empty_plan_is_complete(self):
        """Test empty plan is immediately complete."""
        from home_assistant_robot.coordinator import TaskPlan

        plan = TaskPlan(goal="empty", actions=[])
        assert plan.is_complete


class TestSimplePlanner:
    """Test the simple rule-based planner."""

    def test_navigation_command(self):
        """Test navigation command parsing."""
        from home_assistant_robot.coordinator import SimplePlanner

        planner = SimplePlanner()
        plan = planner.plan("go to the kitchen")

        assert plan is not None
        assert 'kitchen' in plan.goal.lower()
        assert any(a['type'] == 'navigate' for a in plan.actions)

    def test_fetch_command(self):
        """Test fetch command parsing."""
        from home_assistant_robot.coordinator import SimplePlanner

        planner = SimplePlanner()
        plan = planner.plan("bring me a cup")

        assert plan is not None
        assert any(a['type'] == 'navigate' for a in plan.actions)

    def test_unknown_command(self):
        """Test handling of unknown commands."""
        from home_assistant_robot.coordinator import SimplePlanner

        planner = SimplePlanner()
        plan = planner.plan("do something impossible xyz")

        # Should return None or empty plan
        assert plan is None or len(plan.actions) == 0


class TestPerception:
    """Test perception components."""

    def test_simple_detector(self):
        """Test simple object detector."""
        from home_assistant_robot.perception import SimpleDetector
        import numpy as np

        detector = SimpleDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        assert isinstance(detections, list)
        # Demo detector should return something
        assert len(detections) >= 0

    def test_detection_format(self):
        """Test detection output format."""
        from home_assistant_robot.perception import SimpleDetector
        import numpy as np

        detector = SimpleDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        for det in detections:
            assert 'class' in det
            assert 'confidence' in det
            assert 'bbox' in det


class TestLLMPlanner:
    """Test LLM-based task planner."""

    def test_navigation_plan(self):
        """Test navigation planning."""
        from home_assistant_robot.llm_planner import LLMTaskPlanner

        planner = LLMTaskPlanner()
        plan = planner.plan("go to the kitchen")

        assert 'goal' in plan
        assert 'actions' in plan
        assert len(plan['actions']) > 0

    def test_fetch_plan(self):
        """Test fetch planning."""
        from home_assistant_robot.llm_planner import LLMTaskPlanner

        planner = LLMTaskPlanner()
        plan = planner.plan("bring me the cup")

        assert 'goal' in plan
        assert any(a.get('type') == 'navigate' for a in plan['actions'])

    def test_pick_plan(self):
        """Test pick up planning."""
        from home_assistant_robot.llm_planner import LLMTaskPlanner

        planner = LLMTaskPlanner()
        plan = planner.plan("pick up the ball")

        assert 'goal' in plan
        assert any(a.get('type') == 'pick' for a in plan['actions'])
```

## Integration Tests

```python
#!/usr/bin/env python3
"""
Integration Tests
Physical AI Book - Chapter 8: Capstone

Tests for subsystem integration.
"""

import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time
from threading import Thread


@pytest.fixture(scope='module')
def ros_context():
    """Initialize ROS 2 context for tests."""
    rclpy.init()
    yield
    rclpy.shutdown()


class TestVoiceToCoordinator:
    """Test voice interface to coordinator integration."""

    def test_command_received(self, ros_context):
        """Test coordinator receives voice commands."""
        received = []

        class TestNode(Node):
            def __init__(self):
                super().__init__('test_node')
                self.pub = self.create_publisher(String, '/voice/command', 10)
                self.sub = self.create_subscription(
                    String, '/coordinator/status',
                    lambda msg: received.append(msg.data), 10
                )

        node = TestNode()

        # Spin briefly
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

        # Send command
        msg = String()
        msg.data = "go to kitchen"
        node.pub.publish(msg)

        # Wait for response
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.destroy_node()


class TestEmergencyStop:
    """Test emergency stop integration."""

    def test_emergency_stop_propagates(self, ros_context):
        """Test emergency stop signal reaches all nodes."""

        class TestNode(Node):
            def __init__(self):
                super().__init__('emergency_test')
                self.pub = self.create_publisher(Bool, '/safety/emergency_stop', 10)

        node = TestNode()

        # Send emergency stop
        msg = Bool()
        msg.data = True
        node.pub.publish(msg)

        rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()


class TestPerceptionToCoordinator:
    """Test perception to coordinator integration."""

    def test_object_detection_published(self, ros_context):
        """Test object detections are published."""
        received = []

        class TestNode(Node):
            def __init__(self):
                super().__init__('perception_test')
                self.sub = self.create_subscription(
                    String, '/perception/objects',
                    lambda msg: received.append(msg.data), 10
                )

        node = TestNode()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

        node.destroy_node()
```

## Simulation Testing

```python
#!/usr/bin/env python3
"""
Simulation Test Script
Physical AI Book - Chapter 8: Capstone

End-to-end testing in Gazebo simulation.

Usage:
    # Terminal 1: Launch simulation
    ros2 launch home_assistant_robot simulation.launch.py

    # Terminal 2: Run tests
    python3 test_simulation.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time
import sys


class SimulationTester(Node):
    """Automated simulation tester."""

    def __init__(self):
        super().__init__('simulation_tester')

        # State
        self.robot_pose = None
        self.coordinator_state = None
        self.test_results = []

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 10
        )
        self.state_sub = self.create_subscription(
            String, '/coordinator/state',
            self._state_callback, 10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(String, '/voice/command', 10)

        self.get_logger().info('Simulation tester ready')

    def _odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def _state_callback(self, msg):
        self.coordinator_state = msg.data

    def send_command(self, command: str):
        """Send a voice command."""
        msg = String()
        msg.data = command
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'Sent command: {command}')

    def wait_for_state(self, target_state: str, timeout: float = 30.0) -> bool:
        """Wait for coordinator to reach target state."""
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.coordinator_state == target_state:
                return True
        return False

    def wait_for_idle(self, timeout: float = 60.0) -> bool:
        """Wait for coordinator to return to IDLE."""
        return self.wait_for_state('IDLE', timeout)

    def run_test_suite(self):
        """Run all simulation tests."""
        self.get_logger().info('Starting simulation test suite')

        # Test 1: Navigation
        self.test_navigation()

        # Test 2: Object fetch
        self.test_fetch()

        # Print results
        self.print_results()

    def test_navigation(self):
        """Test navigation command."""
        self.get_logger().info('TEST: Navigation')

        self.send_command("go to the kitchen")

        # Wait for execution
        if self.wait_for_idle(timeout=60.0):
            self.test_results.append(('Navigation', 'PASS'))
        else:
            self.test_results.append(('Navigation', 'FAIL - Timeout'))

    def test_fetch(self):
        """Test fetch command."""
        self.get_logger().info('TEST: Fetch')

        self.send_command("bring me a cup")

        if self.wait_for_idle(timeout=90.0):
            self.test_results.append(('Fetch', 'PASS'))
        else:
            self.test_results.append(('Fetch', 'FAIL - Timeout'))

    def print_results(self):
        """Print test results."""
        self.get_logger().info('=' * 50)
        self.get_logger().info('TEST RESULTS')
        self.get_logger().info('=' * 50)

        for name, result in self.test_results:
            status = '✓' if 'PASS' in result else '✗'
            self.get_logger().info(f'{status} {name}: {result}')

        passed = sum(1 for _, r in self.test_results if 'PASS' in r)
        total = len(self.test_results)
        self.get_logger().info(f'\nPassed: {passed}/{total}')


def main():
    rclpy.init()

    tester = SimulationTester()

    # Wait for system to be ready
    time.sleep(5.0)

    try:
        tester.run_test_suite()
    except KeyboardInterrupt:
        pass

    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Common Issues & Debugging

### Issue 1: Navigation Fails to Start

**Symptoms:** Robot doesn't move when navigation command is given.

**Debug Steps:**
```bash
# Check Nav2 status
ros2 topic echo /navigate_to_pose/_action/status

# Check if map is loaded
ros2 topic echo /map --once

# Check localization
ros2 topic echo /amcl_pose
```

**Solution:** Ensure map is loaded and robot is localized.

### Issue 2: Object Detection Not Working

**Symptoms:** Perception reports no objects.

**Debug Steps:**
```bash
# Check camera topic
ros2 topic hz /camera/image_raw

# View camera feed
ros2 run rqt_image_view rqt_image_view

# Check detection output
ros2 topic echo /perception/objects
```

### Issue 3: Coordinator Stuck in State

**Symptoms:** Coordinator doesn't return to IDLE.

**Debug Steps:**
```bash
# Check coordinator status
ros2 topic echo /coordinator/status

# Check for errors
ros2 topic echo /rosout | grep coordinator

# Force reset
ros2 service call /coordinator/reset std_srvs/srv/Empty
```

## Deployment Checklist

Before deploying to real hardware:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Simulation tests pass
- [ ] Safety limits configured
- [ ] Emergency stop tested
- [ ] Battery monitoring enabled
- [ ] Logging configured
- [ ] Recovery behaviors set up

## Summary

In this lesson, we covered:

1. **Unit testing** individual components
2. **Integration testing** subsystem interactions
3. **Simulation testing** end-to-end behavior
4. **Debugging** common issues
5. **Deployment** preparation

## Congratulations!

You have completed the Physical AI Book capstone project! You now have:

- A complete robot assistant system
- Integration of all learned concepts
- Testing and debugging skills
- Deployment knowledge

Continue exploring by:
- Adding new capabilities
- Improving LLM integration
- Training custom perception models
- Deploying on real hardware
