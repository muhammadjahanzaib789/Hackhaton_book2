#!/usr/bin/env python3
"""
LLM Task Planner for Home Assistant Robot
Physical AI Book - Chapter 8: Capstone

Advanced LLM-based task planner that converts natural language
commands into executable action sequences.

Features:
- Multiple LLM provider support (Ollama, OpenAI)
- Context-aware planning with world state
- Multi-step task decomposition
- Error recovery planning
- Safety constraint integration

Usage:
    planner = LLMTaskPlanner()
    plan = planner.plan("bring me a cup from the kitchen")

Dependencies:
    - asyncio
    - aiohttp (for Ollama)
    - openai (optional)

Author: Physical AI Book
License: MIT
"""

import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging


# Configure logging
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Supported action types for the robot."""
    NAVIGATE = "navigate"
    SPEAK = "speak"
    WAIT = "wait"
    STOP = "stop"
    PICK = "pick"
    PLACE = "place"
    FIND_OBJECT = "find_object"
    NAVIGATE_TO_OBJECT = "navigate_to_object"
    LOOK_AT = "look_at"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"


@dataclass
class Action:
    """
    A single action in a task plan.

    Attributes:
        type: The action type
        parameters: Action-specific parameters
        description: Human-readable description
        estimated_duration: Expected time in seconds
        can_fail: Whether this action might fail
        recovery_action: Action to take on failure
    """
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    estimated_duration: float = 1.0
    can_fail: bool = False
    recovery_action: Optional['Action'] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        result = {
            'type': self.type.value,
            **self.parameters,
        }
        if self.description:
            result['description'] = self.description
        return result


@dataclass
class TaskPlan:
    """
    Complete task plan with action sequence.

    Attributes:
        goal: High-level goal description
        actions: Sequence of actions to execute
        confidence: Planner confidence (0.0 to 1.0)
        reasoning: LLM's reasoning for the plan
        alternatives: Alternative approaches if primary fails
    """
    goal: str
    actions: List[Action]
    confidence: float = 1.0
    reasoning: str = ""
    alternatives: List['TaskPlan'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary format."""
        return {
            'goal': self.goal,
            'actions': [a.to_dict() for a in self.actions],
            'confidence': self.confidence,
            'reasoning': self.reasoning,
        }


@dataclass
class WorldState:
    """
    Current state of the robot's world.

    Attributes:
        robot_location: Current robot location name
        robot_pose: Current robot pose (x, y, yaw)
        known_objects: Objects detected and their locations
        held_object: Object currently being held
        battery_level: Battery percentage
        is_charging: Whether robot is charging
    """
    robot_location: str = "home"
    robot_pose: Dict[str, float] = field(default_factory=lambda: {'x': 0, 'y': 0, 'yaw': 0})
    known_objects: Dict[str, Dict] = field(default_factory=dict)
    held_object: Optional[str] = None
    battery_level: float = 100.0
    is_charging: bool = False

    def to_context(self) -> str:
        """Convert world state to context string for LLM."""
        lines = [
            f"Robot is at: {self.robot_location}",
            f"Robot position: ({self.robot_pose['x']:.1f}, {self.robot_pose['y']:.1f})",
            f"Battery: {self.battery_level:.0f}%",
        ]

        if self.held_object:
            lines.append(f"Currently holding: {self.held_object}")

        if self.known_objects:
            lines.append("Known objects:")
            for obj, info in self.known_objects.items():
                loc = info.get('location', 'unknown')
                lines.append(f"  - {obj} at {loc}")

        return "\n".join(lines)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local inference."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate response using Ollama API."""
        try:
            import aiohttp

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        logger.error(f"Ollama error: {response.status}")
                        return ""
        except ImportError:
            logger.error("aiohttp not installed")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception:
            return False


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and demos."""

    async def generate(self, prompt: str, system: str = "") -> str:
        """Return mock response based on prompt content."""
        # This is used when no real LLM is available
        return "{}"

    async def is_available(self) -> bool:
        """Mock is always available."""
        return True


class LLMTaskPlanner:
    """
    LLM-based task planner for natural language commands.

    Converts voice commands into executable action sequences
    using LLM reasoning and world state context.

    Example:
        planner = LLMTaskPlanner()
        plan = planner.plan("go to the kitchen and find a cup")
    """

    # Environment knowledge
    LOCATIONS = {
        'kitchen': {'x': 5.0, 'y': 2.0, 'yaw': 0.0},
        'living room': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        'bedroom': {'x': -3.0, 'y': 4.0, 'yaw': 1.57},
        'bathroom': {'x': -3.0, 'y': -2.0, 'yaw': 3.14},
        'home': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        'charging station': {'x': 0.0, 'y': -1.0, 'yaw': 0.0},
        'front door': {'x': 8.0, 'y': 0.0, 'yaw': 0.0},
        'dining room': {'x': 3.0, 'y': 4.0, 'yaw': 0.0},
    }

    KNOWN_OBJECTS = [
        'cup', 'bottle', 'ball', 'remote', 'book', 'phone',
        'keys', 'wallet', 'glasses', 'medicine', 'snack'
    ]

    SYSTEM_PROMPT = """You are a robot task planner. Convert user commands into action sequences.

Available actions:
- navigate: Move to a location (target: {x, y, yaw}, location_name: string)
- speak: Say something (message: string)
- wait: Wait for duration (duration: seconds)
- stop: Stop all motion
- pick: Pick up an object (object: string)
- place: Place held object (position: {x, y, z})
- find_object: Search for an object (object: string)
- look_at: Turn to face something (target: string or {x, y})

Available locations: {locations}
Known objects: {objects}

Respond with JSON only:
{{
    "goal": "brief description of the goal",
    "actions": [
        {{"type": "action_type", "param1": "value1", ...}},
        ...
    ],
    "reasoning": "brief explanation of the plan"
}}"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        use_llm: bool = True
    ):
        """
        Initialize the task planner.

        Args:
            llm_provider: LLM provider instance (auto-detects if None)
            use_llm: Whether to use LLM (falls back to rules if False)
        """
        self.llm_provider = llm_provider
        self.use_llm = use_llm
        self.world_state = WorldState()

        # Try to auto-detect provider
        if self.llm_provider is None and self.use_llm:
            self.llm_provider = OllamaProvider()

    def update_world_state(self, state: WorldState) -> None:
        """Update the world state for context-aware planning."""
        self.world_state = state

    def plan(self, command: str, world_state: Optional[WorldState] = None) -> Dict[str, Any]:
        """
        Generate an action plan from a natural language command.

        Args:
            command: Natural language command from user
            world_state: Current world state (uses stored state if None)

        Returns:
            Plan dictionary with goal and actions
        """
        if world_state:
            self.world_state = world_state

        # Try LLM planning first
        if self.use_llm and self.llm_provider:
            try:
                plan = asyncio.get_event_loop().run_until_complete(
                    self._llm_plan(command)
                )
                if plan and plan.get('actions'):
                    return plan
            except RuntimeError:
                # No event loop running
                try:
                    plan = asyncio.run(self._llm_plan(command))
                    if plan and plan.get('actions'):
                        return plan
                except Exception as e:
                    logger.warning(f"LLM planning failed: {e}")
            except Exception as e:
                logger.warning(f"LLM planning failed: {e}")

        # Fall back to rule-based planning
        return self._rule_based_plan(command)

    async def _llm_plan(self, command: str) -> Optional[Dict[str, Any]]:
        """Use LLM to generate a plan."""
        if not self.llm_provider:
            return None

        # Check availability
        if not await self.llm_provider.is_available():
            logger.warning("LLM provider not available")
            return None

        # Build prompt
        system = self.SYSTEM_PROMPT.format(
            locations=list(self.LOCATIONS.keys()),
            objects=self.KNOWN_OBJECTS
        )

        prompt = f"""Current state:
{self.world_state.to_context()}

User command: "{command}"

Generate an action plan:"""

        # Get response
        response = await self.llm_provider.generate(prompt, system)

        if not response:
            return None

        # Parse JSON response
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                plan_dict = json.loads(response[json_start:json_end])
                return self._validate_plan(plan_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")

        return None

    def _validate_plan(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and normalize a plan dictionary."""
        if not isinstance(plan, dict):
            return None

        if 'goal' not in plan or 'actions' not in plan:
            return None

        if not isinstance(plan['actions'], list):
            return None

        # Validate each action
        valid_actions = []
        for action in plan['actions']:
            if not isinstance(action, dict):
                continue

            action_type = action.get('type', '')

            # Validate action type
            try:
                ActionType(action_type)
                valid_actions.append(action)
            except ValueError:
                logger.warning(f"Unknown action type: {action_type}")
                continue

        if not valid_actions:
            return None

        return {
            'goal': str(plan['goal']),
            'actions': valid_actions,
            'reasoning': plan.get('reasoning', ''),
        }

    def _rule_based_plan(self, command: str) -> Dict[str, Any]:
        """
        Generate plan using rule-based approach.

        This is the fallback when LLM is not available.
        """
        cmd = command.lower().strip()

        # === Navigation commands ===
        for loc_name, loc_pose in self.LOCATIONS.items():
            if loc_name in cmd:
                if any(word in cmd for word in ['go', 'navigate', 'move', 'come', 'take me']):
                    return {
                        'goal': f'Navigate to {loc_name}',
                        'actions': [
                            {'type': 'speak', 'message': f'Heading to the {loc_name}'},
                            {
                                'type': 'navigate',
                                'target': loc_pose,
                                'location_name': loc_name
                            },
                            {'type': 'speak', 'message': f'I have arrived at the {loc_name}'}
                        ],
                        'reasoning': f'User requested navigation to {loc_name}'
                    }

        # === Fetch/bring commands ===
        if any(word in cmd for word in ['bring', 'fetch', 'get', 'grab']):
            # Try to identify the object
            target_object = None
            for obj in self.KNOWN_OBJECTS:
                if obj in cmd:
                    target_object = obj
                    break

            if target_object:
                return self._create_fetch_plan(target_object)
            else:
                return {
                    'goal': 'Ask for clarification',
                    'actions': [
                        {'type': 'speak', 'message': 'What would you like me to get for you?'}
                    ],
                    'reasoning': 'Object not specified in fetch command'
                }

        # === Pick up commands ===
        if 'pick' in cmd and ('up' in cmd or 'it' in cmd):
            for obj in self.KNOWN_OBJECTS:
                if obj in cmd:
                    return {
                        'goal': f'Pick up {obj}',
                        'actions': [
                            {'type': 'speak', 'message': f'Looking for the {obj}'},
                            {'type': 'find_object', 'object': obj},
                            {'type': 'navigate_to_object', 'object': obj},
                            {'type': 'pick', 'object': obj},
                            {'type': 'speak', 'message': f'I have picked up the {obj}'}
                        ],
                        'reasoning': f'User requested to pick up {obj}'
                    }

        # === Stop commands ===
        if any(word in cmd for word in ['stop', 'halt', 'freeze', 'cancel']):
            return {
                'goal': 'Stop all motion',
                'actions': [
                    {'type': 'stop'},
                    {'type': 'speak', 'message': 'Stopped.'}
                ],
                'reasoning': 'User requested stop'
            }

        # === Come here / follow commands ===
        if any(phrase in cmd for phrase in ['come here', 'come to me', 'follow me']):
            return {
                'goal': 'Come to user',
                'actions': [
                    {'type': 'speak', 'message': 'Coming to you'},
                    {'type': 'navigate', 'target': self.LOCATIONS['home'], 'location_name': 'user'},
                    {'type': 'speak', 'message': 'I am here'}
                ],
                'reasoning': 'User requested robot to come'
            }

        # === Greeting commands ===
        if any(word in cmd for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return {
                'goal': 'Greet user',
                'actions': [
                    {'type': 'speak', 'message': 'Hello! How can I help you today?'}
                ],
                'reasoning': 'User greeted the robot'
            }

        # === Help commands ===
        if any(word in cmd for word in ['help', 'what can you do', 'capabilities']):
            return {
                'goal': 'Explain capabilities',
                'actions': [
                    {
                        'type': 'speak',
                        'message': 'I can navigate to rooms, fetch objects, and help around the house. '
                                   'Try saying: go to the kitchen, or bring me a cup.'
                    }
                ],
                'reasoning': 'User requested help'
            }

        # === Status commands ===
        if any(word in cmd for word in ['status', 'battery', 'how are you']):
            return {
                'goal': 'Report status',
                'actions': [
                    {
                        'type': 'speak',
                        'message': f'I am at {self.world_state.robot_location}. '
                                   f'My battery is at {self.world_state.battery_level:.0f} percent.'
                    }
                ],
                'reasoning': 'User requested status'
            }

        # === Go home / return commands ===
        if any(phrase in cmd for phrase in ['go home', 'return home', 'go back', 'go to base']):
            return {
                'goal': 'Return home',
                'actions': [
                    {'type': 'speak', 'message': 'Returning to home base'},
                    {'type': 'navigate', 'target': self.LOCATIONS['home'], 'location_name': 'home'},
                    {'type': 'speak', 'message': 'I am back home'}
                ],
                'reasoning': 'User requested return to home'
            }

        # === Charge commands ===
        if any(word in cmd for word in ['charge', 'charging', 'dock']):
            return {
                'goal': 'Go to charging station',
                'actions': [
                    {'type': 'speak', 'message': 'Going to charge'},
                    {'type': 'navigate', 'target': self.LOCATIONS['charging station'],
                     'location_name': 'charging station'},
                    {'type': 'speak', 'message': 'Docked at charging station'}
                ],
                'reasoning': 'User requested charging'
            }

        # === Unknown command ===
        return {
            'goal': 'Unknown',
            'actions': [
                {
                    'type': 'speak',
                    'message': "I'm not sure how to help with that. "
                               "Could you try rephrasing your request?"
                }
            ],
            'reasoning': 'Command not recognized'
        }

    def _create_fetch_plan(self, obj: str) -> Dict[str, Any]:
        """Create a plan to fetch an object."""
        # Determine where to look for the object
        search_location = 'kitchen'  # Default search location

        # Check if object location is known
        if obj in self.world_state.known_objects:
            obj_info = self.world_state.known_objects[obj]
            if 'location' in obj_info:
                search_location = obj_info['location']

        return {
            'goal': f'Fetch {obj}',
            'actions': [
                {'type': 'speak', 'message': f'I will get the {obj} for you'},
                {
                    'type': 'navigate',
                    'target': self.LOCATIONS.get(search_location, self.LOCATIONS['kitchen']),
                    'location_name': search_location
                },
                {'type': 'find_object', 'object': obj},
                {'type': 'pick', 'object': obj},
                {
                    'type': 'navigate',
                    'target': self.LOCATIONS['home'],
                    'location_name': 'user'
                },
                {'type': 'speak', 'message': f'Here is the {obj}'}
            ],
            'reasoning': f'Fetching {obj} from {search_location}'
        }

    def get_recovery_plan(self, failed_action: Dict[str, Any], error: str) -> Optional[Dict[str, Any]]:
        """
        Generate a recovery plan for a failed action.

        Args:
            failed_action: The action that failed
            error: Error description

        Returns:
            Recovery plan or None if no recovery possible
        """
        action_type = failed_action.get('type', '')

        if action_type == 'navigate':
            # Try alternative route or ask for help
            return {
                'goal': 'Recover from navigation failure',
                'actions': [
                    {'type': 'speak', 'message': f'Navigation failed: {error}'},
                    {'type': 'speak', 'message': 'Let me try a different approach'},
                    {'type': 'navigate', 'target': self.LOCATIONS['home'], 'location_name': 'home'},
                ],
                'reasoning': 'Returning home after navigation failure'
            }

        elif action_type == 'pick':
            return {
                'goal': 'Recover from pick failure',
                'actions': [
                    {'type': 'speak', 'message': f'I could not pick that up: {error}'},
                    {'type': 'speak', 'message': 'Could you please hand it to me?'},
                ],
                'reasoning': 'Asking for human assistance after pick failure'
            }

        elif action_type == 'find_object':
            obj = failed_action.get('object', 'the object')
            return {
                'goal': 'Recover from object search failure',
                'actions': [
                    {'type': 'speak', 'message': f'I could not find the {obj}'},
                    {'type': 'speak', 'message': 'Could you tell me where it is?'},
                ],
                'reasoning': 'Asking for location help after search failure'
            }

        # Generic recovery
        return {
            'goal': 'Report failure',
            'actions': [
                {'type': 'speak', 'message': f'I encountered an error: {error}'},
                {'type': 'stop'},
            ],
            'reasoning': 'Stopping and reporting after unrecoverable error'
        }


def main():
    """Demo the task planner."""
    import sys

    # Create planner (will use rule-based if LLM not available)
    planner = LLMTaskPlanner(use_llm=False)  # Use rules for demo

    # Test commands
    test_commands = [
        "go to the kitchen",
        "bring me a cup",
        "pick up the ball",
        "hello",
        "what can you do",
        "stop",
        "go home",
    ]

    print("=" * 60)
    print("LLM Task Planner Demo")
    print("=" * 60)

    for cmd in test_commands:
        print(f"\nCommand: \"{cmd}\"")
        print("-" * 40)

        plan = planner.plan(cmd)

        print(f"Goal: {plan['goal']}")
        print("Actions:")
        for i, action in enumerate(plan['actions'], 1):
            print(f"  {i}. {action['type']}: {action}")

        if plan.get('reasoning'):
            print(f"Reasoning: {plan['reasoning']}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == '__main__':
    main()
