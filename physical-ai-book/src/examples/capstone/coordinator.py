#!/usr/bin/env python3
"""
Home Assistant Robot Coordinator
Physical AI Book - Chapter 8: Capstone

Central coordinator node that orchestrates all robot subsystems
using a state machine architecture.

Features:
- State machine for task management
- Voice command processing
- LLM-based task planning
- Action execution via subsystems
- Safety monitoring
- Error recovery

Usage:
    ros2 run home_assistant_robot coordinator

Dependencies:
    - rclpy
    - nav2_msgs
    - std_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import BatteryState
from nav2_msgs.action import NavigateToPose
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import json
import threading
import time


class RobotState(Enum):
    """
    Robot coordinator states.

    State transitions:
    IDLE -> LISTENING -> PLANNING -> EXECUTING -> IDLE
    Any state -> EMERGENCY_STOP
    Any state -> ERROR -> IDLE
    """
    IDLE = auto()
    LISTENING = auto()
    PLANNING = auto()
    CONFIRMING = auto()
    EXECUTING = auto()
    PAUSED = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()


@dataclass
class TaskPlan:
    """
    Task plan with action sequence.

    Attributes:
        goal: Human-readable goal description
        actions: List of action dictionaries
        current_index: Current action being executed
        status: Plan status (pending, executing, complete, failed)
        metadata: Additional plan information
    """
    goal: str
    actions: List[Dict[str, Any]]
    current_index: int = 0
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_action(self) -> Optional[Dict[str, Any]]:
        """Get current action being executed."""
        if 0 <= self.current_index < len(self.actions):
            return self.actions[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all actions have been executed."""
        return self.current_index >= len(self.actions)

    @property
    def progress(self) -> float:
        """Get execution progress (0.0 to 1.0)."""
        if not self.actions:
            return 1.0
        return self.current_index / len(self.actions)

    def advance(self) -> None:
        """Move to next action."""
        self.current_index += 1

    def reset(self) -> None:
        """Reset plan execution."""
        self.current_index = 0
        self.status = "pending"


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

    # Known locations in the environment
    LOCATIONS = {
        'kitchen': {'x': 5.0, 'y': 2.0, 'yaw': 0.0},
        'living room': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        'bedroom': {'x': -3.0, 'y': 4.0, 'yaw': 1.57},
        'bathroom': {'x': -3.0, 'y': -2.0, 'yaw': 3.14},
        'home': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        'charging station': {'x': 0.0, 'y': -1.0, 'yaw': 0.0},
    }

    def __init__(self):
        super().__init__('coordinator')

        # Parameters
        self.declare_parameter('enable_voice', True)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('min_battery_level', 20.0)
        self.declare_parameter('max_task_duration', 300.0)

        self.enable_voice = self.get_parameter('enable_voice').value
        self.safety_enabled = self.get_parameter('safety_enabled').value
        self.min_battery = self.get_parameter('min_battery_level').value
        self.max_duration = self.get_parameter('max_task_duration').value

        # State
        self.state = RobotState.IDLE
        self.current_plan: Optional[TaskPlan] = None
        self.battery_level = 100.0
        self.task_start_time: Optional[float] = None

        # Thread safety
        self._lock = threading.Lock()

        # Callback groups
        self._sensor_cb = ReentrantCallbackGroup()
        self._action_cb = MutuallyExclusiveCallbackGroup()

        # === Subscribers ===
        self.voice_sub = self.create_subscription(
            String, '/voice/command',
            self._voice_callback, 10
        )
        self.emergency_sub = self.create_subscription(
            Bool, '/safety/emergency_stop',
            self._emergency_callback, 10
        )
        self.battery_sub = self.create_subscription(
            BatteryState, '/battery_state',
            self._battery_callback, 10,
            callback_group=self._sensor_cb
        )

        # === Publishers ===
        self.status_pub = self.create_publisher(String, '/coordinator/status', 10)
        self.state_pub = self.create_publisher(String, '/coordinator/state', 10)
        self.speech_pub = self.create_publisher(String, '/speech/say', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # === Action Clients ===
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self._action_cb
        )

        # === Timers ===
        self.status_timer = self.create_timer(1.0, self._publish_status)
        self.watchdog_timer = self.create_timer(5.0, self._watchdog)

        # === Initialize planner ===
        self.planner = SimplePlanner(self.LOCATIONS)

        # Action handlers
        self._action_handlers = {
            'navigate': self._execute_navigation,
            'speak': self._execute_speak,
            'wait': self._execute_wait,
            'stop': self._execute_stop,
        }

        # Current action state
        self._current_goal_handle = None
        self._action_complete_callback: Optional[Callable] = None

        self.get_logger().info('Coordinator initialized')
        self._speak("Hello! I'm ready to help you.")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _voice_callback(self, msg: String):
        """Handle voice commands."""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Voice command: "{command}"')

        with self._lock:
            # Check if we can accept commands
            if self.state == RobotState.EMERGENCY_STOP:
                self._speak("Emergency stop is active. Please reset the system.")
                return

            if self.state not in (RobotState.IDLE, RobotState.LISTENING):
                self._speak("I'm currently busy. Please wait.")
                return

            # Check battery
            if self.battery_level < self.min_battery:
                self._speak(f"My battery is low at {self.battery_level:.0f}%. I need to charge.")
                return

            self._transition_to(RobotState.PLANNING)

        # Plan and execute
        self._plan_and_execute(command)

    def _emergency_callback(self, msg: Bool):
        """Handle emergency stop signals."""
        if msg.data:
            self.get_logger().error('EMERGENCY STOP ACTIVATED')
            self._stop_all_motion()
            self._transition_to(RobotState.EMERGENCY_STOP)
            self._speak("Emergency stop activated. All motion stopped.")
        else:
            if self.state == RobotState.EMERGENCY_STOP:
                self.get_logger().info('Emergency stop cleared')
                self._transition_to(RobotState.IDLE)
                self._speak("Emergency stop cleared. Ready for commands.")

    def _battery_callback(self, msg: BatteryState):
        """Update battery level."""
        self.battery_level = msg.percentage * 100

        # Low battery warning
        if self.battery_level < self.min_battery and self.state == RobotState.EXECUTING:
            self.get_logger().warn(f'Low battery: {self.battery_level:.0f}%')
            self._speak("Warning: Low battery. Returning home to charge.")
            self._cancel_current_task()
            # Would navigate to charging station

    # =========================================================================
    # Planning
    # =========================================================================

    def _plan_and_execute(self, command: str):
        """Plan task and begin execution."""
        try:
            # Generate plan
            plan = self.planner.plan(command)

            if plan is None or not plan.actions:
                self._speak("I'm not sure how to help with that. Could you rephrase?")
                self._transition_to(RobotState.IDLE)
                return

            self.current_plan = plan
            self.task_start_time = time.time()

            self._speak(f"I'll {plan.goal}. Starting now.")
            self.get_logger().info(f'Executing plan: {plan.goal} ({len(plan.actions)} actions)')

            # Begin execution
            self._transition_to(RobotState.EXECUTING)
            self._execute_next_action()

        except Exception as e:
            self.get_logger().error(f'Planning error: {e}')
            self._speak("Sorry, I had trouble understanding that request.")
            self._transition_to(RobotState.IDLE)

    # =========================================================================
    # Execution
    # =========================================================================

    def _execute_next_action(self):
        """Execute the next action in the plan."""
        if self.state != RobotState.EXECUTING:
            return

        if self.current_plan is None or self.current_plan.is_complete:
            self._task_complete()
            return

        action = self.current_plan.current_action
        if action is None:
            self._task_complete()
            return

        action_type = action.get('type', 'unknown')
        self.get_logger().info(
            f'Action [{self.current_plan.current_index + 1}/'
            f'{len(self.current_plan.actions)}]: {action_type}'
        )

        # Get handler
        handler = self._action_handlers.get(action_type)
        if handler:
            handler(action)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            self._action_complete(True)

    def _action_complete(self, success: bool, error: str = ""):
        """Handle action completion."""
        if not success:
            self.get_logger().error(f'Action failed: {error}')
            self._speak(f"I couldn't complete that step. {error}")
            self.current_plan.status = "failed"
            self._transition_to(RobotState.IDLE)
            return

        # Advance to next action
        self.current_plan.advance()
        self._execute_next_action()

    def _task_complete(self):
        """Handle task completion."""
        if self.current_plan:
            self.current_plan.status = "complete"
            duration = time.time() - (self.task_start_time or time.time())
            self.get_logger().info(
                f'Task complete: {self.current_plan.goal} '
                f'({duration:.1f}s)'
            )

        self._speak("Done! Is there anything else I can help with?")
        self.current_plan = None
        self.task_start_time = None
        self._transition_to(RobotState.IDLE)

    def _cancel_current_task(self):
        """Cancel the current task."""
        if self._current_goal_handle:
            self._current_goal_handle.cancel_goal_async()

        self.current_plan = None
        self._transition_to(RobotState.IDLE)

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _execute_navigation(self, action: Dict):
        """Execute navigation action."""
        target = action.get('target', {})
        location_name = action.get('location_name', 'target')

        # Wait for server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self._action_complete(False, "Navigation server unavailable")
            return

        # Create goal
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(target.get('x', 0))
        goal.pose.pose.position.y = float(target.get('y', 0))

        import math
        yaw = float(target.get('yaw', 0))
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        self._speak(f"Navigating to the {location_name}")

        # Send goal
        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self._nav_feedback
        )
        future.add_done_callback(self._nav_goal_response)

    def _nav_feedback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        remaining = feedback.distance_remaining
        self.get_logger().debug(f'Navigation: {remaining:.2f}m remaining')

    def _nav_goal_response(self, future):
        """Handle navigation goal response."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self._action_complete(False, "Navigation goal rejected")
            return

        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result)

    def _nav_result(self, future):
        """Handle navigation result."""
        result = future.result()
        self._current_goal_handle = None

        if result.status == 4:  # SUCCEEDED
            self._action_complete(True)
        else:
            self._action_complete(False, f"Navigation failed (status={result.status})")

    def _execute_speak(self, action: Dict):
        """Execute speak action."""
        message = action.get('message', '')
        self._speak(message)
        # Small delay for speech
        self.create_timer(
            0.5, lambda: self._action_complete(True),
            callback_group=self._action_cb
        )

    def _execute_wait(self, action: Dict):
        """Execute wait action."""
        duration = float(action.get('duration', 1.0))
        self.get_logger().debug(f'Waiting {duration}s')
        self.create_timer(
            duration, lambda: self._action_complete(True),
            callback_group=self._action_cb
        )

    def _execute_stop(self, action: Dict):
        """Execute stop action."""
        self._stop_all_motion()
        self._action_complete(True)

    # =========================================================================
    # Utilities
    # =========================================================================

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
        self.get_logger().info(f'Speech: "{text}"')

    def _stop_all_motion(self):
        """Stop all robot motion."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def _publish_status(self):
        """Publish coordinator status."""
        status = {
            'state': self.state.name,
            'battery': self.battery_level,
            'current_task': self.current_plan.goal if self.current_plan else None,
            'progress': self.current_plan.progress if self.current_plan else 0.0,
            'action_index': self.current_plan.current_index if self.current_plan else 0,
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _watchdog(self):
        """Watchdog timer for stuck tasks."""
        if self.state == RobotState.EXECUTING and self.task_start_time:
            elapsed = time.time() - self.task_start_time
            if elapsed > self.max_duration:
                self.get_logger().warn(f'Task timeout after {elapsed:.0f}s')
                self._speak("This task is taking too long. Stopping.")
                self._cancel_current_task()


class SimplePlanner:
    """Simple rule-based task planner."""

    def __init__(self, locations: Dict[str, Dict]):
        self.locations = locations

    def plan(self, command: str) -> Optional[TaskPlan]:
        """Generate plan from command."""
        cmd = command.lower()

        # Navigation commands
        for loc_name, loc_pose in self.locations.items():
            if loc_name in cmd and any(w in cmd for w in ['go', 'navigate', 'move', 'come']):
                return TaskPlan(
                    goal=f"go to the {loc_name}",
                    actions=[
                        {'type': 'speak', 'message': f'Heading to the {loc_name}'},
                        {'type': 'navigate', 'location_name': loc_name, 'target': loc_pose},
                        {'type': 'speak', 'message': f'I have arrived at the {loc_name}'}
                    ]
                )

        # Stop command
        if 'stop' in cmd or 'halt' in cmd:
            return TaskPlan(
                goal="stop",
                actions=[
                    {'type': 'stop'},
                    {'type': 'speak', 'message': 'Stopped.'}
                ]
            )

        # Fetch commands
        if any(w in cmd for w in ['bring', 'fetch', 'get']):
            return TaskPlan(
                goal="fetch the requested item",
                actions=[
                    {'type': 'speak', 'message': 'I will get that for you'},
                    {'type': 'navigate', 'location_name': 'kitchen',
                     'target': self.locations.get('kitchen', {'x': 0, 'y': 0, 'yaw': 0})},
                    {'type': 'speak', 'message': 'Looking for the item'},
                    {'type': 'wait', 'duration': 2.0},
                    {'type': 'navigate', 'location_name': 'home',
                     'target': self.locations.get('home', {'x': 0, 'y': 0, 'yaw': 0})},
                    {'type': 'speak', 'message': 'Here you go!'}
                ]
            )

        # Hello/greeting
        if any(w in cmd for w in ['hello', 'hi', 'hey']):
            return TaskPlan(
                goal="greet user",
                actions=[
                    {'type': 'speak', 'message': 'Hello! How can I help you today?'}
                ]
            )

        return None


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = CoordinatorNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
