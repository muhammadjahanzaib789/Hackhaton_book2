#!/usr/bin/env python3
"""
LLM Task Planner Node
Physical AI Book - Chapter 6: LLM Integration

ROS 2 node that uses an LLM to plan and decompose tasks
into executable robot actions.

Usage:
    ros2 run physical_ai_examples llm_task_planner

    # Send a task request
    ros2 topic pub /task/request std_msgs/String "data: 'pick up the red cup'"

Expected Output:
    [INFO] [llm_task_planner]: Task planner ready (using ollama)
    [INFO] [llm_task_planner]: Received task: "pick up the red cup"
    [INFO] [llm_task_planner]: Generated plan with 5 actions
    [INFO] [llm_task_planner]:   1. scan_area - Look for the red cup
    [INFO] [llm_task_planner]:   2. navigate_to - Approach the cup
    [INFO] [llm_task_planner]:   3. open_gripper - Prepare to grasp
    [INFO] [llm_task_planner]:   4. pick_object - Pick up the cup
    [INFO] [llm_task_planner]:   5. speak - Confirm completion

Dependencies:
    - rclpy
    - requests (for LLM API calls)

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os


class ActionType(Enum):
    """Robot action categories."""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    SYSTEM = "system"


@dataclass
class PlannedAction:
    """A single planned action."""
    name: str
    params: Dict[str, Any]
    description: str
    action_type: ActionType


@dataclass
class TaskPlan:
    """Complete task plan."""
    task: str
    actions: List[PlannedAction]
    estimated_time: float
    preconditions: List[str]


# Available robot actions for the LLM to use
AVAILABLE_ACTIONS = """
AVAILABLE ROBOT ACTIONS:
- navigate_to(target: str) - Navigate to a location
- approach_object(object_id: str) - Move close to an object
- pick_object(object_id: str) - Pick up an object
- place_object(target: str) - Place held object at location
- open_gripper() - Open the gripper
- close_gripper() - Close the gripper
- scan_area() - Look around to find objects
- look_at(target: str) - Look at a specific object or location
- speak(message: str) - Say something to the user
- wait(duration: float) - Wait for some time
"""


class LLMTaskPlannerNode(Node):
    """
    ROS 2 node for LLM-based task planning.

    Converts natural language commands into executable action sequences.
    """

    SYSTEM_PROMPT = """You are a robot task planner. Convert user commands into action sequences.

{actions}

RULES:
1. Only use actions from the AVAILABLE list
2. Consider safety - never rush near humans
3. Verify object presence before manipulation
4. Announce actions before executing

Output JSON format:
{{
  "understanding": "Brief task description",
  "actions": [
    {{"action": "action_name", "params": {{}}, "description": "why"}}
  ],
  "preconditions": ["required conditions"],
  "estimated_time": <seconds>
}}"""

    def __init__(self):
        super().__init__('llm_task_planner')

        # Parameters
        self.declare_parameter('llm_provider', 'ollama')
        self.declare_parameter('model', 'llama3')
        self.declare_parameter('ollama_host', 'http://localhost:11434')

        self.provider = self.get_parameter('llm_provider').value
        self.model = self.get_parameter('model').value
        self.ollama_host = self.get_parameter('ollama_host').value

        # World state (in production, from perception)
        self.world_state = {
            "robot_position": "living_room",
            "objects": [
                {"id": "red_cup", "type": "cup", "color": "red", "location": "table"},
                {"id": "blue_book", "type": "book", "color": "blue", "location": "shelf"},
            ],
            "gripper_state": "empty"
        }

        # Subscribers
        self.create_subscription(
            String, '/task/request',
            self._task_request_callback, 10
        )

        # Publishers
        self.plan_pub = self.create_publisher(String, '/task/plan', 10)
        self.action_pub = self.create_publisher(String, '/task/action', 10)
        self.status_pub = self.create_publisher(String, '/task/status', 10)

        self.get_logger().info(f'Task planner ready (using {self.provider})')

    def _task_request_callback(self, msg: String):
        """Handle incoming task requests."""
        task = msg.data.strip()
        self.get_logger().info(f'Received task: "{task}"')

        # Generate plan
        plan = self._generate_plan(task)

        if plan:
            self.get_logger().info(f'Generated plan with {len(plan.actions)} actions')

            for i, action in enumerate(plan.actions):
                self.get_logger().info(
                    f'  {i+1}. {action.name} - {action.description}'
                )

            # Publish plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'task': plan.task,
                'actions': [
                    {'name': a.name, 'params': a.params, 'description': a.description}
                    for a in plan.actions
                ],
                'estimated_time': plan.estimated_time
            })
            self.plan_pub.publish(plan_msg)

            # Execute plan
            self._execute_plan(plan)
        else:
            self.get_logger().error('Failed to generate plan')
            status_msg = String()
            status_msg.data = 'failed'
            self.status_pub.publish(status_msg)

    def _generate_plan(self, task: str) -> Optional[TaskPlan]:
        """Generate a task plan using LLM."""
        prompt = self._build_prompt(task)

        try:
            response = self._call_llm(prompt)
            return self._parse_plan(task, response)
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return None

    def _build_prompt(self, task: str) -> str:
        """Build the planning prompt."""
        system = self.SYSTEM_PROMPT.format(actions=AVAILABLE_ACTIONS)

        return f"""{system}

CURRENT STATE:
{json.dumps(self.world_state, indent=2)}

USER TASK: {task}

JSON plan:"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        import requests

        if self.provider == 'ollama':
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['response']

        elif self.provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError('OPENAI_API_KEY not set')

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1024
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        else:
            raise ValueError(f'Unknown provider: {self.provider}')

    def _parse_plan(self, task: str, response: str) -> Optional[TaskPlan]:
        """Parse LLM response into TaskPlan."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start < 0:
                return None

            data = json.loads(response[json_start:json_end])

            actions = []
            for action_data in data.get('actions', []):
                action_name = action_data.get('action', '')

                # Determine action type
                if action_name in ['navigate_to', 'approach_object']:
                    action_type = ActionType.NAVIGATION
                elif action_name in ['pick_object', 'place_object', 'open_gripper', 'close_gripper']:
                    action_type = ActionType.MANIPULATION
                elif action_name in ['scan_area', 'look_at']:
                    action_type = ActionType.PERCEPTION
                elif action_name == 'speak':
                    action_type = ActionType.COMMUNICATION
                else:
                    action_type = ActionType.SYSTEM

                actions.append(PlannedAction(
                    name=action_name,
                    params=action_data.get('params', {}),
                    description=action_data.get('description', ''),
                    action_type=action_type
                ))

            return TaskPlan(
                task=task,
                actions=actions,
                estimated_time=data.get('estimated_time', 30.0),
                preconditions=data.get('preconditions', [])
            )

        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f'Failed to parse plan: {e}')
            return None

    def _execute_plan(self, plan: TaskPlan):
        """Execute the planned actions."""
        self.get_logger().info('Executing plan...')

        for i, action in enumerate(plan.actions):
            self.get_logger().info(
                f'Executing action {i+1}/{len(plan.actions)}: {action.name}'
            )

            # Publish current action
            action_msg = String()
            action_msg.data = json.dumps({
                'action': action.name,
                'params': action.params,
                'index': i,
                'total': len(plan.actions)
            })
            self.action_pub.publish(action_msg)

            # Simulate action execution
            success = self._execute_action(action)

            if not success:
                self.get_logger().error(f'Action {action.name} failed')
                status_msg = String()
                status_msg.data = 'failed'
                self.status_pub.publish(status_msg)
                return

        self.get_logger().info('Plan executed successfully')
        status_msg = String()
        status_msg.data = 'complete'
        self.status_pub.publish(status_msg)

    def _execute_action(self, action: PlannedAction) -> bool:
        """Execute a single action (simulated)."""
        import time

        # Simulate action execution
        durations = {
            'navigate_to': 2.0,
            'approach_object': 1.0,
            'pick_object': 1.5,
            'place_object': 1.5,
            'open_gripper': 0.5,
            'close_gripper': 0.5,
            'scan_area': 1.0,
            'look_at': 0.5,
            'speak': 1.0,
            'wait': action.params.get('duration', 1.0)
        }

        duration = durations.get(action.name, 1.0)
        time.sleep(duration)

        return True  # Simulate success


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = LLMTaskPlannerNode()

    # Demo: Process a sample task
    import time
    time.sleep(1.0)

    demo_msg = String()
    demo_msg.data = "pick up the red cup from the table"
    node._task_request_callback(demo_msg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
