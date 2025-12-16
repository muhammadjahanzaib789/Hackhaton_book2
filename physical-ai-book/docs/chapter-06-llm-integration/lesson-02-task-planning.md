---
sidebar_position: 2
title: "Lesson 2: LLM-Based Task Planning"
description: "Using language models for task decomposition and planning"
---

# LLM-Based Task Planning

## Learning Objectives

By the end of this lesson, you will be able to:

1. Decompose high-level commands into robot primitives
2. Define action schemas for robot capabilities
3. Implement a task planning pipeline
4. Handle planning failures gracefully

## Prerequisites

- Completed Lesson 1 (LLM Basics)
- Understanding of robot action primitives
- Familiarity with state machines

## Task Planning Overview

```
┌─────────────────────────────────────────────────────────────┐
│              LLM Task Planning Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User: "Make me a coffee"                                   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Intent Recognition (LLM)                      │   │
│  │   → Task: prepare_beverage                          │   │
│  │   → Item: coffee                                    │   │
│  │   → Recipient: user                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Task Decomposition (LLM)                      │   │
│  │   1. Navigate to kitchen                            │   │
│  │   2. Locate coffee machine                          │   │
│  │   3. Check if cup available                         │   │
│  │   4. Place cup under dispenser                      │   │
│  │   5. Press brew button                              │   │
│  │   6. Wait for brewing                               │   │
│  │   7. Pick up cup                                    │   │
│  │   8. Navigate to user                               │   │
│  │   9. Hand over cup                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Action Grounding                              │   │
│  │   Map to robot primitives with parameters           │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Execution & Monitoring                        │   │
│  │   Execute actions, handle failures                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Action Schema Definition

### Robot Primitives

```python
"""
Robot Action Schemas
Physical AI Book - Chapter 6

Defines available robot actions and their parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json


class ActionType(Enum):
    """Categories of robot actions."""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    SYSTEM = "system"


@dataclass
class ActionParameter:
    """Definition of an action parameter."""
    name: str
    type: str  # 'string', 'number', 'boolean', 'object'
    description: str
    required: bool = True
    default: Any = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class ActionSchema:
    """Schema defining a robot action."""
    name: str
    description: str
    action_type: ActionType
    parameters: List[ActionParameter]
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    duration_estimate: float = 1.0  # seconds

    def to_prompt_description(self) -> str:
        """Generate description for LLM prompt."""
        params_desc = ", ".join([
            f"{p.name}: {p.type}" + (" (required)" if p.required else "")
            for p in self.parameters
        ])
        return f"{self.name}({params_desc}) - {self.description}"

    def validate_params(self, params: Dict[str, Any]) -> tuple:
        """
        Validate parameters against schema.

        Returns:
            (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

            if param.name in params:
                value = params[param.name]

                # Type checking
                if param.type == 'number':
                    if not isinstance(value, (int, float)):
                        return False, f"{param.name} must be a number"

                    # Constraint checking
                    if param.constraints:
                        if 'min' in param.constraints:
                            if value < param.constraints['min']:
                                return False, f"{param.name} below minimum"
                        if 'max' in param.constraints:
                            if value > param.constraints['max']:
                                return False, f"{param.name} above maximum"

                elif param.type == 'string':
                    if not isinstance(value, str):
                        return False, f"{param.name} must be a string"

                    if param.constraints and 'enum' in param.constraints:
                        if value not in param.constraints['enum']:
                            return False, f"{param.name} must be one of {param.constraints['enum']}"

        return True, None


# Define standard robot actions
ROBOT_ACTIONS = {
    # Navigation
    "navigate_to": ActionSchema(
        name="navigate_to",
        description="Navigate robot to a named location or coordinates",
        action_type=ActionType.NAVIGATION,
        parameters=[
            ActionParameter("target", "string", "Target location name or 'x,y' coordinates"),
            ActionParameter("speed", "number", "Navigation speed (0.1-1.0)", required=False,
                          default=0.5, constraints={"min": 0.1, "max": 1.0}),
        ],
        preconditions=["localized", "path_clear"],
        effects=["at_location(target)"],
        duration_estimate=10.0
    ),

    "approach_object": ActionSchema(
        name="approach_object",
        description="Move within manipulation distance of an object",
        action_type=ActionType.NAVIGATION,
        parameters=[
            ActionParameter("object_id", "string", "ID of object to approach"),
            ActionParameter("approach_distance", "number", "Distance to stop at", required=False,
                          default=0.3, constraints={"min": 0.2, "max": 1.0}),
        ],
        preconditions=["object_visible(object_id)"],
        effects=["near_object(object_id)"],
        duration_estimate=5.0
    ),

    # Manipulation
    "pick_object": ActionSchema(
        name="pick_object",
        description="Pick up an object with the gripper",
        action_type=ActionType.MANIPULATION,
        parameters=[
            ActionParameter("object_id", "string", "ID of object to pick"),
            ActionParameter("grasp_type", "string", "Type of grasp", required=False,
                          default="top", constraints={"enum": ["top", "side", "pinch"]}),
        ],
        preconditions=["near_object(object_id)", "gripper_empty", "object_graspable(object_id)"],
        effects=["holding(object_id)", "not gripper_empty"],
        duration_estimate=8.0
    ),

    "place_object": ActionSchema(
        name="place_object",
        description="Place held object at a location",
        action_type=ActionType.MANIPULATION,
        parameters=[
            ActionParameter("target", "string", "Target location or surface"),
            ActionParameter("offset", "object", "Position offset {x, y, z}", required=False),
        ],
        preconditions=["not gripper_empty"],
        effects=["gripper_empty", "object_at(held_object, target)"],
        duration_estimate=6.0
    ),

    "open_gripper": ActionSchema(
        name="open_gripper",
        description="Open the gripper to release object",
        action_type=ActionType.MANIPULATION,
        parameters=[
            ActionParameter("width", "number", "Target opening width", required=False,
                          default=0.08, constraints={"min": 0, "max": 0.1}),
        ],
        preconditions=[],
        effects=["gripper_empty"],
        duration_estimate=1.0
    ),

    "close_gripper": ActionSchema(
        name="close_gripper",
        description="Close gripper to grasp object",
        action_type=ActionType.MANIPULATION,
        parameters=[
            ActionParameter("force", "number", "Grasp force", required=False,
                          default=20.0, constraints={"min": 5, "max": 50}),
        ],
        preconditions=[],
        effects=[],
        duration_estimate=1.0
    ),

    "press_button": ActionSchema(
        name="press_button",
        description="Press a button or switch",
        action_type=ActionType.MANIPULATION,
        parameters=[
            ActionParameter("button_id", "string", "ID of button to press"),
            ActionParameter("duration", "number", "Press duration in seconds", required=False,
                          default=0.5, constraints={"min": 0.1, "max": 3.0}),
        ],
        preconditions=["near_object(button_id)"],
        effects=["button_pressed(button_id)"],
        duration_estimate=2.0
    ),

    # Perception
    "look_at": ActionSchema(
        name="look_at",
        description="Turn head/camera to look at target",
        action_type=ActionType.PERCEPTION,
        parameters=[
            ActionParameter("target", "string", "Target to look at (object_id or location)"),
        ],
        preconditions=[],
        effects=["looking_at(target)"],
        duration_estimate=1.0
    ),

    "scan_area": ActionSchema(
        name="scan_area",
        description="Scan surroundings to detect objects",
        action_type=ActionType.PERCEPTION,
        parameters=[
            ActionParameter("target_type", "string", "Type of object to find", required=False),
        ],
        preconditions=[],
        effects=["area_scanned"],
        duration_estimate=3.0
    ),

    "identify_object": ActionSchema(
        name="identify_object",
        description="Identify a specific object in view",
        action_type=ActionType.PERCEPTION,
        parameters=[
            ActionParameter("description", "string", "Description of object to find"),
        ],
        preconditions=[],
        effects=["object_identified"],
        duration_estimate=2.0
    ),

    # Communication
    "speak": ActionSchema(
        name="speak",
        description="Speak a message to the user",
        action_type=ActionType.COMMUNICATION,
        parameters=[
            ActionParameter("message", "string", "Message to speak"),
            ActionParameter("language", "string", "Language code", required=False, default="en"),
        ],
        preconditions=[],
        effects=["message_spoken"],
        duration_estimate=3.0
    ),

    "wait_for_response": ActionSchema(
        name="wait_for_response",
        description="Wait for user verbal response",
        action_type=ActionType.COMMUNICATION,
        parameters=[
            ActionParameter("timeout", "number", "Max wait time in seconds", required=False,
                          default=30.0, constraints={"min": 5, "max": 120}),
            ActionParameter("expected_responses", "object", "List of expected responses", required=False),
        ],
        preconditions=[],
        effects=["response_received"],
        duration_estimate=10.0
    ),

    # System
    "wait": ActionSchema(
        name="wait",
        description="Wait for specified duration",
        action_type=ActionType.SYSTEM,
        parameters=[
            ActionParameter("duration", "number", "Wait duration in seconds",
                          constraints={"min": 0.1, "max": 300}),
        ],
        preconditions=[],
        effects=[],
        duration_estimate=5.0
    ),

    "check_condition": ActionSchema(
        name="check_condition",
        description="Check a condition before proceeding",
        action_type=ActionType.SYSTEM,
        parameters=[
            ActionParameter("condition", "string", "Condition to check"),
        ],
        preconditions=[],
        effects=["condition_checked"],
        duration_estimate=1.0
    ),
}


def get_actions_prompt() -> str:
    """Generate actions description for LLM prompt."""
    lines = ["AVAILABLE ACTIONS:"]
    for action in ROBOT_ACTIONS.values():
        lines.append(f"  - {action.to_prompt_description()}")
    return "\n".join(lines)
```

## Task Decomposition

### LLM-Based Planner

```python
"""
LLM Task Planner
Physical AI Book - Chapter 6

Decomposes high-level tasks into robot primitives.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class PlannedAction:
    """A planned robot action."""
    action_name: str
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float


@dataclass
class TaskPlan:
    """Complete plan for a task."""
    task_description: str
    actions: List[PlannedAction]
    total_duration: float
    preconditions: List[str]
    success_criteria: List[str]


class TaskPlanner:
    """
    LLM-based task planner.

    Converts natural language commands into executable plans.
    """

    PLANNING_PROMPT = """You are a robot task planner. Convert user commands into action sequences.

{actions_description}

WORLD STATE:
{world_state}

PLANNING RULES:
1. Only use actions from the AVAILABLE ACTIONS list
2. Check preconditions before each action
3. Include error checking steps
4. Minimize unnecessary movements
5. Consider human safety at all times

USER COMMAND: {command}

Respond with a JSON plan:
{{
  "understanding": "What the user wants",
  "preconditions": ["Things that must be true before starting"],
  "actions": [
    {{"action": "action_name", "params": {{}}, "description": "Why this action"}}
  ],
  "success_criteria": ["How to verify task completion"],
  "estimated_time": <seconds>
}}

JSON plan:"""

    def __init__(self, llm_provider):
        """
        Initialize planner.

        Args:
            llm_provider: LLM provider for planning
        """
        self.llm = llm_provider
        self.action_schemas = ROBOT_ACTIONS

    def plan(self, command: str, world_state: Dict) -> Optional[TaskPlan]:
        """
        Generate a plan for the given command.

        Args:
            command: Natural language command
            world_state: Current world state

        Returns:
            TaskPlan or None if planning fails
        """
        # Build prompt
        prompt = self.PLANNING_PROMPT.format(
            actions_description=get_actions_prompt(),
            world_state=json.dumps(world_state, indent=2),
            command=command
        )

        # Get LLM response
        response = self.llm.generate(prompt)

        # Parse response
        plan = self._parse_plan(response.content)
        if plan is None:
            return None

        # Validate plan
        is_valid, errors = self._validate_plan(plan)
        if not is_valid:
            print(f"Plan validation failed: {errors}")
            return None

        return plan

    def _parse_plan(self, response: str) -> Optional[TaskPlan]:
        """Parse LLM response into TaskPlan."""
        try:
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1:
                return None

            data = json.loads(response[json_start:json_end])

            # Build plan
            actions = []
            for action_data in data.get('actions', []):
                action_name = action_data['action']

                # Get duration estimate from schema
                schema = self.action_schemas.get(action_name)
                duration = schema.duration_estimate if schema else 1.0

                actions.append(PlannedAction(
                    action_name=action_name,
                    parameters=action_data.get('params', {}),
                    description=action_data.get('description', ''),
                    estimated_duration=duration
                ))

            return TaskPlan(
                task_description=data.get('understanding', ''),
                actions=actions,
                total_duration=data.get('estimated_time', sum(a.estimated_duration for a in actions)),
                preconditions=data.get('preconditions', []),
                success_criteria=data.get('success_criteria', [])
            )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse plan: {e}")
            return None

    def _validate_plan(self, plan: TaskPlan) -> tuple:
        """Validate a plan against action schemas."""
        errors = []

        for i, action in enumerate(plan.actions):
            # Check action exists
            if action.action_name not in self.action_schemas:
                errors.append(f"Action {i}: Unknown action '{action.action_name}'")
                continue

            # Validate parameters
            schema = self.action_schemas[action.action_name]
            is_valid, error = schema.validate_params(action.parameters)
            if not is_valid:
                errors.append(f"Action {i} ({action.action_name}): {error}")

        return len(errors) == 0, errors

    def replan_on_failure(self, original_plan: TaskPlan,
                         failed_action: PlannedAction,
                         failure_reason: str,
                         world_state: Dict) -> Optional[TaskPlan]:
        """
        Generate new plan after action failure.

        Args:
            original_plan: The original plan
            failed_action: Action that failed
            failure_reason: Why it failed
            world_state: Current world state

        Returns:
            New TaskPlan or None
        """
        replan_prompt = f"""Previous plan failed during execution.

Original task: {original_plan.task_description}
Failed action: {failed_action.action_name}
Failure reason: {failure_reason}

Current world state: {json.dumps(world_state)}

Generate a recovery plan to complete the original task, accounting for the failure.
If task cannot be completed, respond with {{"cannot_complete": true, "reason": "explanation"}}

{get_actions_prompt()}

JSON recovery plan:"""

        response = self.llm.generate(replan_prompt)
        return self._parse_plan(response.content)
```

## Task Executor

### Execution Engine

```python
"""
Task Executor
Physical AI Book - Chapter 6

Executes planned actions with monitoring and recovery.
"""

import rclpy
from rclpy.node import Node
from typing import Callable, Dict, Any, Optional
from enum import Enum
import time


class ExecutionStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionResult:
    """Result of executing an action."""
    def __init__(self, success: bool, message: str = "",
                 data: Optional[Dict[str, Any]] = None):
        self.success = success
        self.message = message
        self.data = data or {}


class TaskExecutor(Node):
    """
    Executes task plans with monitoring and recovery.

    Interfaces with robot subsystems to execute actions.
    """

    def __init__(self, planner: TaskPlanner):
        super().__init__('task_executor')

        self.planner = planner

        # Action handlers map action names to execution functions
        self._action_handlers: Dict[str, Callable] = {}

        # Current execution state
        self._current_plan: Optional[TaskPlan] = None
        self._current_action_idx: int = 0
        self._status = ExecutionStatus.PENDING

        # Callbacks
        self._on_action_complete: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None

        self.get_logger().info('Task executor initialized')

    def register_action_handler(self, action_name: str,
                                handler: Callable[[Dict], ActionResult]):
        """
        Register handler for an action type.

        Args:
            action_name: Name of action
            handler: Function(params) -> ActionResult
        """
        self._action_handlers[action_name] = handler
        self.get_logger().debug(f'Registered handler for {action_name}')

    def execute_plan(self, plan: TaskPlan) -> ExecutionStatus:
        """
        Execute a task plan.

        Args:
            plan: TaskPlan to execute

        Returns:
            Final execution status
        """
        self._current_plan = plan
        self._current_action_idx = 0
        self._status = ExecutionStatus.RUNNING

        self.get_logger().info(f'Executing plan: {plan.task_description}')
        self.get_logger().info(f'Total actions: {len(plan.actions)}')

        while self._current_action_idx < len(plan.actions):
            if self._status == ExecutionStatus.CANCELLED:
                break

            action = plan.actions[self._current_action_idx]
            self.get_logger().info(
                f'Action {self._current_action_idx + 1}/{len(plan.actions)}: '
                f'{action.action_name}'
            )

            # Execute action
            result = self._execute_action(action)

            if result.success:
                self._current_action_idx += 1

                if self._on_action_complete:
                    self._on_action_complete(action, result)
            else:
                self.get_logger().error(
                    f'Action failed: {action.action_name} - {result.message}'
                )

                # Attempt recovery
                if not self._attempt_recovery(action, result.message):
                    self._status = ExecutionStatus.FAILED
                    break

        if self._status == ExecutionStatus.RUNNING:
            self._status = ExecutionStatus.SUCCESS

        if self._on_task_complete:
            self._on_task_complete(plan, self._status)

        return self._status

    def _execute_action(self, action: PlannedAction) -> ActionResult:
        """Execute a single action."""
        handler = self._action_handlers.get(action.action_name)

        if handler is None:
            return ActionResult(
                success=False,
                message=f"No handler for action: {action.action_name}"
            )

        try:
            self.get_logger().debug(
                f'Executing {action.action_name} with params: {action.parameters}'
            )
            return handler(action.parameters)

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Exception during execution: {str(e)}"
            )

    def _attempt_recovery(self, failed_action: PlannedAction,
                         failure_reason: str) -> bool:
        """
        Attempt to recover from action failure.

        Args:
            failed_action: The action that failed
            failure_reason: Why it failed

        Returns:
            True if recovery succeeded
        """
        self.get_logger().info('Attempting recovery...')

        # Get current world state
        world_state = self._get_world_state()

        # Ask planner for recovery plan
        recovery_plan = self.planner.replan_on_failure(
            self._current_plan,
            failed_action,
            failure_reason,
            world_state
        )

        if recovery_plan is None:
            self.get_logger().error('No recovery plan generated')
            return False

        self.get_logger().info(
            f'Recovery plan has {len(recovery_plan.actions)} actions'
        )

        # Replace remaining actions with recovery plan
        self._current_plan.actions = (
            self._current_plan.actions[:self._current_action_idx] +
            recovery_plan.actions
        )

        return True

    def _get_world_state(self) -> Dict[str, Any]:
        """Get current world state for replanning."""
        # In production, would query perception system
        return {
            "robot_position": "unknown",
            "held_object": None,
            "visible_objects": [],
            "error_state": True
        }

    def cancel(self):
        """Cancel current execution."""
        self._status = ExecutionStatus.CANCELLED
        self.get_logger().info('Execution cancelled')


# Example action handlers
def handle_navigate_to(params: Dict) -> ActionResult:
    """Handle navigate_to action."""
    target = params.get('target')
    print(f"  [NAV] Navigating to {target}")
    time.sleep(1.0)  # Simulate navigation
    return ActionResult(success=True)


def handle_pick_object(params: Dict) -> ActionResult:
    """Handle pick_object action."""
    object_id = params.get('object_id')
    print(f"  [MANIP] Picking up {object_id}")
    time.sleep(0.5)
    return ActionResult(success=True, data={"held_object": object_id})


def handle_place_object(params: Dict) -> ActionResult:
    """Handle place_object action."""
    target = params.get('target')
    print(f"  [MANIP] Placing object at {target}")
    time.sleep(0.5)
    return ActionResult(success=True)


def handle_speak(params: Dict) -> ActionResult:
    """Handle speak action."""
    message = params.get('message')
    print(f"  [SPEAK] Robot says: {message}")
    return ActionResult(success=True)


def handle_wait(params: Dict) -> ActionResult:
    """Handle wait action."""
    duration = params.get('duration', 1.0)
    print(f"  [WAIT] Waiting {duration}s")
    time.sleep(duration)
    return ActionResult(success=True)
```

## Complete Integration Example

```python
#!/usr/bin/env python3
"""
LLM Task Planning Demo
Physical AI Book - Chapter 6

Complete example of LLM-based task planning and execution.
"""

def main():
    # Initialize LLM provider (using Ollama for local inference)
    config = LLMConfig(
        model="llama3",
        temperature=0.3,  # Lower for more deterministic planning
        max_tokens=2048,
        system_prompt=ROBOT_SYSTEM_PROMPT
    )

    llm = LLMProvider.create("ollama", config)

    # Initialize planner
    planner = TaskPlanner(llm)

    # Initialize executor with ROS node
    rclpy.init()
    executor = TaskExecutor(planner)

    # Register action handlers
    executor.register_action_handler("navigate_to", handle_navigate_to)
    executor.register_action_handler("pick_object", handle_pick_object)
    executor.register_action_handler("place_object", handle_place_object)
    executor.register_action_handler("speak", handle_speak)
    executor.register_action_handler("wait", handle_wait)

    # Current world state
    world_state = {
        "robot_position": "living_room",
        "visible_objects": [
            {"id": "cup_1", "type": "cup", "color": "red", "location": "table"},
            {"id": "book_1", "type": "book", "location": "shelf"},
        ],
        "gripper_state": "empty",
        "battery_level": 85
    }

    # Example commands
    commands = [
        "Bring me the red cup from the table",
        "Put the book on the desk",
        "Go to the kitchen and tell me what you see",
    ]

    for command in commands:
        print(f"\n{'='*60}")
        print(f"Command: {command}")
        print('='*60)

        # Generate plan
        plan = planner.plan(command, world_state)

        if plan is None:
            print("Failed to generate plan")
            continue

        print(f"\nPlan: {plan.task_description}")
        print(f"Estimated time: {plan.total_duration:.1f}s")
        print(f"Actions:")
        for i, action in enumerate(plan.actions):
            print(f"  {i+1}. {action.action_name}({action.parameters})")
            print(f"      - {action.description}")

        # Execute plan
        print(f"\nExecuting...")
        status = executor.execute_plan(plan)
        print(f"Result: {status.value}")

    executor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **Action schemas** define robot capabilities formally
2. **Task decomposition** converts commands to primitives
3. **Plan validation** ensures executable plans
4. **Recovery planning** handles failures gracefully
5. **Execution monitoring** tracks progress

## Next Steps

In the [next lesson](./lesson-03-voice-interaction.md), we will:
- Add speech recognition and synthesis
- Build a complete voice-controlled interface
- Handle multi-turn conversations

## Additional Resources

- [Task and Motion Planning](https://arxiv.org/abs/2010.01083)
- [SayCan: Grounding Language in Robotic Affordances](https://say-can.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
