---
sidebar_position: 2
title: "Lesson 2: Core Implementation"
description: "Building the core components of the Physical AI Assistant"
---

# Core Implementation

## Overview

In this lesson, we implement the core components of our Physical AI Assistant:

1. **Coordinator Node** - Central state machine
2. **LLM Planner** - Task understanding and planning
3. **Action Clients** - Interface to subsystems
4. **Safety Monitor** - Safety oversight

## System State Machine

```
┌─────────────────────────────────────────────────────────────┐
│              Coordinator State Machine                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────┐                            │
│                      │  IDLE   │                            │
│                      └────┬────┘                            │
│                           │ voice_command                   │
│                           ▼                                 │
│                    ┌─────────────┐                          │
│                    │  PLANNING   │                          │
│                    └──────┬──────┘                          │
│                           │ plan_ready                      │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │      EXECUTING         │                     │
│              │  ┌──────────────────┐  │                     │
│              │  │ Current Action:  │  │                     │
│              │  │  - Navigate      │  │                     │
│              │  │  - Perceive      │  │                     │
│              │  │  - Manipulate    │  │                     │
│              │  └──────────────────┘  │                     │
│              └───────────┬────────────┘                     │
│                          │                                  │
│            ┌─────────────┼─────────────┐                   │
│            │             │             │                    │
│            ▼             ▼             ▼                    │
│      ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│      │ SUCCESS  │  │  FAILED  │  │ CANCELED │             │
│      └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│           │             │             │                    │
│           └─────────────┴─────────────┘                    │
│                         │                                  │
│                         ▼                                  │
│                    ┌─────────┐                             │
│                    │  IDLE   │                             │
│                    └─────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Coordinator Node Implementation

```python
#!/usr/bin/env python3
"""
Home Assistant Robot Coordinator
Physical AI Book - Chapter 8: Capstone

Central coordinator that orchestrates all robot subsystems
using a state machine architecture.

Usage:
    ros2 run home_assistant_robot coordinator

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import threading


class RobotState(Enum):
    """Robot coordinator states."""
    IDLE = auto()
    LISTENING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING_CONFIRMATION = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()


@dataclass
class TaskPlan:
    """Planned task with action sequence."""
    goal: str
    actions: List[Dict[str, Any]]
    current_index: int = 0
    status: str = "pending"

    @property
    def current_action(self) -> Optional[Dict[str, Any]]:
        if self.current_index < len(self.actions):
            return self.actions[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.actions)

    def advance(self):
        self.current_index += 1


class CoordinatorNode(Node):
    """
    Central coordinator for the home assistant robot.

    Manages:
    - Voice command processing
    - LLM-based task planning
    - Action execution via subsystems
    - Safety monitoring
    - Error recovery
    """

    def __init__(self):
        super().__init__('coordinator')

        # State
        self.state = RobotState.IDLE
        self.current_plan: Optional[TaskPlan] = None
        self._lock = threading.Lock()

        # Callback group for concurrent operations
        self._cb_group = ReentrantCallbackGroup()

        # === Subscribers ===
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice/command',
            self._voice_command_callback, 10
        )
        self.emergency_sub = self.create_subscription(
            Bool, '/safety/emergency_stop',
            self._emergency_callback, 10
        )

        # === Publishers ===
        self.status_pub = self.create_publisher(String, '/coordinator/status', 10)
        self.speech_pub = self.create_publisher(String, '/speech/say', 10)
        self.state_pub = self.create_publisher(String, '/coordinator/state', 10)

        # === Action Clients ===
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self._cb_group
        )

        # === Timers ===
        self.status_timer = self.create_timer(1.0, self._publish_status)

        # === Initialize subsystems ===
        self._init_planner()

        self.get_logger().info('Coordinator initialized')
        self._speak("Hello! I'm ready to help.")

    def _init_planner(self):
        """Initialize LLM planner."""
        # In production, would initialize actual LLM
        self.planner = SimplePlanner()

    def _voice_command_callback(self, msg: String):
        """Handle voice commands."""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Received command: "{command}"')

        with self._lock:
            if self.state == RobotState.EMERGENCY_STOP:
                self._speak("Emergency stop is active. Please reset first.")
                return

            if self.state != RobotState.IDLE:
                self._speak("I'm currently busy. Please wait.")
                return

            self._transition_to(RobotState.PLANNING)

        # Plan in background
        self._plan_and_execute(command)

    def _plan_and_execute(self, command: str):
        """Plan task and begin execution."""
        try:
            # Generate plan
            plan = self.planner.plan(command)

            if not plan or not plan.actions:
                self._speak("I'm not sure how to do that.")
                self._transition_to(RobotState.IDLE)
                return

            self.current_plan = plan
            self._speak(f"I'll {plan.goal}. Starting now.")

            # Begin execution
            self._transition_to(RobotState.EXECUTING)
            self._execute_next_action()

        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            self._speak("Sorry, I encountered an error while planning.")
            self._transition_to(RobotState.IDLE)

    def _execute_next_action(self):
        """Execute the next action in the plan."""
        if self.current_plan is None or self.current_plan.is_complete:
            self._complete_task()
            return

        action = self.current_plan.current_action
        action_type = action.get('type', '')

        self.get_logger().info(f'Executing: {action_type}')

        if action_type == 'navigate':
            self._execute_navigation(action)
        elif action_type == 'speak':
            self._execute_speak(action)
        elif action_type == 'wait':
            self._execute_wait(action)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            self._action_complete(success=True)

    def _execute_navigation(self, action: Dict):
        """Execute navigation action."""
        target = action.get('target', {})

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(target.get('x', 0))
        goal.pose.pose.position.y = float(target.get('y', 0))
        goal.pose.pose.orientation.w = 1.0

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            self._action_complete(success=False, error="Navigation unavailable")
            return

        self._speak(f"Navigating to {action.get('location', 'target')}")

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self._nav_feedback
        )
        future.add_done_callback(self._nav_goal_response)

    def _nav_feedback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        remaining = feedback.distance_remaining
        self.get_logger().debug(f'Distance remaining: {remaining:.2f}m')

    def _nav_goal_response(self, future):
        """Handle navigation goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._action_complete(success=False, error="Navigation rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result)

    def _nav_result(self, future):
        """Handle navigation result."""
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self._action_complete(success=True)
        else:
            self._action_complete(success=False, error="Navigation failed")

    def _execute_speak(self, action: Dict):
        """Execute speak action."""
        message = action.get('message', '')
        self._speak(message)
        self._action_complete(success=True)

    def _execute_wait(self, action: Dict):
        """Execute wait action."""
        duration = action.get('duration', 1.0)
        self.create_timer(
            duration,
            lambda: self._action_complete(success=True),
            callback_group=self._cb_group
        )

    def _action_complete(self, success: bool, error: str = ""):
        """Handle action completion."""
        if not success:
            self.get_logger().error(f'Action failed: {error}')
            self._speak(f"I couldn't complete that step. {error}")
            self._transition_to(RobotState.IDLE)
            return

        self.current_plan.advance()
        self._execute_next_action()

    def _complete_task(self):
        """Complete the current task."""
        self._speak("Task completed successfully!")
        self.current_plan = None
        self._transition_to(RobotState.IDLE)

    def _emergency_callback(self, msg: Bool):
        """Handle emergency stop."""
        if msg.data:
            self.get_logger().error('EMERGENCY STOP ACTIVATED')
            self._transition_to(RobotState.EMERGENCY_STOP)
            self._speak("Emergency stop activated.")

    def _transition_to(self, new_state: RobotState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.get_logger().info(f'State: {old_state.name} -> {new_state.name}')

        # Publish state change
        msg = String()
        msg.data = new_state.name
        self.state_pub.publish(msg)

    def _speak(self, text: str):
        """Send text to speech system."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def _publish_status(self):
        """Publish coordinator status."""
        status = {
            'state': self.state.name,
            'plan': self.current_plan.goal if self.current_plan else None,
            'action_index': self.current_plan.current_index if self.current_plan else 0
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


class SimplePlanner:
    """Simple rule-based planner for demo purposes."""

    LOCATIONS = {
        'kitchen': {'x': 5.0, 'y': 2.0},
        'living room': {'x': 0.0, 'y': 0.0},
        'bedroom': {'x': -3.0, 'y': 4.0},
        'home': {'x': 0.0, 'y': 0.0},
    }

    def plan(self, command: str) -> Optional[TaskPlan]:
        """Generate a simple plan from command."""
        command_lower = command.lower()

        # Navigation commands
        for location, coords in self.LOCATIONS.items():
            if location in command_lower and ('go' in command_lower or 'navigate' in command_lower):
                return TaskPlan(
                    goal=f"go to the {location}",
                    actions=[
                        {'type': 'speak', 'message': f'Heading to the {location}'},
                        {'type': 'navigate', 'location': location, 'target': coords},
                        {'type': 'speak', 'message': f'I have arrived at the {location}'}
                    ]
                )

        # Fetch commands
        if 'bring' in command_lower or 'fetch' in command_lower or 'get' in command_lower:
            return TaskPlan(
                goal="fetch the requested item",
                actions=[
                    {'type': 'speak', 'message': 'I will get that for you'},
                    {'type': 'navigate', 'location': 'kitchen', 'target': self.LOCATIONS['kitchen']},
                    {'type': 'speak', 'message': 'Looking for the item'},
                    {'type': 'wait', 'duration': 2.0},
                    {'type': 'navigate', 'location': 'home', 'target': self.LOCATIONS['home']},
                    {'type': 'speak', 'message': 'Here you go!'}
                ]
            )

        return None


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Message Definitions

### Task.msg

```
# Task message for coordinator
string task_id
string instruction
string status
string[] actions
float64 progress
```

### SystemStatus.msg

```
# System status message
string state
string current_task
float64 battery_level
bool emergency_stop
string[] active_subsystems
```

## Summary

In this lesson, we implemented:

1. **State machine** for robot coordination
2. **Coordinator node** with action clients
3. **Simple planner** for task decomposition
4. **Message types** for communication

## Next Steps

Continue to [Lesson 3](./lesson-03-integration.md) to:
- Integrate perception pipeline
- Add manipulation capabilities
- Implement full LLM planning
