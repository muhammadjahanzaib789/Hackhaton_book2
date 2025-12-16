#!/usr/bin/env python3
"""
Action Schema Definitions
Physical AI Book - Chapter 6: LLM Integration

Defines structured schemas for LLM-to-robot action translation.
These schemas ensure type safety and validation between
natural language understanding and robot execution.

Design:
- ActionSchema: Defines valid robot actions
- ActionResponse: LLM output for action planning
- RobotAction: Executable robot command

Usage:
    from physical_ai_examples.llm.core import ActionSchema, RobotAction

    # Define allowed actions
    schema = ActionSchema(
        name="pick_object",
        parameters={
            "object_id": {"type": "string", "required": True},
            "position": {"type": "array", "items": "number"}
        }
    )

    # Parse LLM response into action
    action = RobotAction.from_llm_response(llm_output)

Author: Physical AI Book
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class ActionType(Enum):
    """
    Supported robot action types.

    These map to ROS 2 action servers in the capstone system.
    """
    # Navigation actions
    NAVIGATE_TO = "navigate_to"
    FOLLOW_PATH = "follow_path"
    STOP = "stop"

    # Manipulation actions
    PICK = "pick"
    PLACE = "place"
    GRASP = "grasp"
    RELEASE = "release"
    MOVE_ARM = "move_arm"

    # Vision actions
    FIND_OBJECT = "find_object"
    TRACK_OBJECT = "track_object"
    SCAN_AREA = "scan_area"

    # Speech actions
    SPEAK = "speak"
    LISTEN = "listen"

    # Composite actions
    FETCH = "fetch"
    DELIVER = "deliver"

    # System actions
    WAIT = "wait"
    ABORT = "abort"


@dataclass
class ActionParameter:
    """
    Definition of an action parameter.

    Attributes:
        name: Parameter name
        param_type: Data type (string, number, array, object)
        required: Whether parameter is required
        default: Default value if not provided
        constraints: Validation constraints
        description: Human-readable description
    """
    name: str
    param_type: str
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def validate(self, value: Any) -> bool:
        """
        Validate a value against this parameter definition.

        Args:
            value: Value to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        if value is None:
            if self.required and self.default is None:
                raise ValueError(f"Required parameter '{self.name}' is missing")
            return True

        # Type validation
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        expected = type_map.get(self.param_type)
        if expected and not isinstance(value, expected):
            raise ValueError(
                f"Parameter '{self.name}' expected {self.param_type}, "
                f"got {type(value).__name__}"
            )

        # Constraint validation
        if 'min' in self.constraints and value < self.constraints['min']:
            raise ValueError(
                f"Parameter '{self.name}' below minimum {self.constraints['min']}"
            )
        if 'max' in self.constraints and value > self.constraints['max']:
            raise ValueError(
                f"Parameter '{self.name}' above maximum {self.constraints['max']}"
            )
        if 'enum' in self.constraints and value not in self.constraints['enum']:
            raise ValueError(
                f"Parameter '{self.name}' must be one of {self.constraints['enum']}"
            )
        if 'pattern' in self.constraints:
            import re
            if not re.match(self.constraints['pattern'], str(value)):
                raise ValueError(
                    f"Parameter '{self.name}' doesn't match pattern"
                )

        return True


@dataclass
class ActionSchema:
    """
    Schema defining a valid robot action.

    Attributes:
        name: Action name (must be ActionType value)
        description: Human-readable description
        parameters: Dictionary of parameter definitions
        timeout: Default timeout in seconds
        preconditions: Conditions that must be true
        effects: Expected effects of the action
    """
    name: str
    description: str = ""
    parameters: Dict[str, ActionParameter] = field(default_factory=dict)
    timeout: float = 30.0
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against this schema.

        Args:
            params: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        for name, param_def in self.parameters.items():
            value = params.get(name, param_def.default)
            param_def.validate(value)

        # Check for unknown parameters
        for name in params:
            if name not in self.parameters:
                raise ValueError(f"Unknown parameter '{name}'")

        return True

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM prompting."""
        properties = {}
        required = []

        for name, param in self.parameters.items():
            properties[name] = {
                'type': param.param_type,
                'description': param.description,
            }
            if param.constraints:
                properties[name].update(param.constraints)
            if param.required:
                required.append(name)

        return {
            'type': 'object',
            'properties': properties,
            'required': required,
            'description': self.description,
        }


@dataclass
class RobotAction:
    """
    An executable robot action.

    This is what gets sent to ROS 2 action servers.

    Attributes:
        action_type: Type of action to perform
        parameters: Action parameters
        priority: Execution priority (higher = more urgent)
        timeout: Action timeout in seconds
        metadata: Additional information
    """
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobotAction':
        """
        Create a RobotAction from a dictionary.

        Args:
            data: Dictionary with action data

        Returns:
            RobotAction instance
        """
        action_type = ActionType(data['action_type'])
        return cls(
            action_type=action_type,
            parameters=data.get('parameters', {}),
            priority=data.get('priority', 0),
            timeout=data.get('timeout', 30.0),
            metadata=data.get('metadata', {}),
        )

    @classmethod
    def from_llm_response(cls, response: str) -> 'RobotAction':
        """
        Parse an LLM response into a RobotAction.

        Args:
            response: JSON string from LLM

        Returns:
            RobotAction instance

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            data = json.loads(response)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'priority': self.priority,
            'timeout': self.timeout,
            'metadata': self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ActionResponse:
    """
    Complete action planning response from LLM.

    Contains a sequence of actions to achieve a goal.

    Attributes:
        goal: Original goal/intent
        plan: List of actions to execute
        reasoning: LLM's reasoning for the plan
        confidence: Confidence score (0-1)
        alternatives: Alternative plans if primary fails
    """
    goal: str
    plan: List[RobotAction]
    reasoning: str = ""
    confidence: float = 0.0
    alternatives: List[List[RobotAction]] = field(default_factory=list)

    @classmethod
    def from_llm_output(cls, output: Dict[str, Any]) -> 'ActionResponse':
        """
        Parse LLM output into ActionResponse.

        Expected format:
        {
            "goal": "fetch the red cup",
            "plan": [
                {"action_type": "find_object", "parameters": {"object": "red cup"}},
                {"action_type": "navigate_to", "parameters": {"target": "object"}},
                {"action_type": "pick", "parameters": {"object": "red cup"}}
            ],
            "reasoning": "First locate the cup, then navigate and pick it up",
            "confidence": 0.85
        }

        Args:
            output: Parsed JSON from LLM

        Returns:
            ActionResponse instance
        """
        plan = [RobotAction.from_dict(a) for a in output.get('plan', [])]
        alternatives = [
            [RobotAction.from_dict(a) for a in alt]
            for alt in output.get('alternatives', [])
        ]

        return cls(
            goal=output.get('goal', ''),
            plan=plan,
            reasoning=output.get('reasoning', ''),
            confidence=output.get('confidence', 0.0),
            alternatives=alternatives,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'goal': self.goal,
            'plan': [a.to_dict() for a in self.plan],
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'alternatives': [
                [a.to_dict() for a in alt]
                for alt in self.alternatives
            ],
        }


# Pre-defined action schemas for common robot actions
STANDARD_SCHEMAS = {
    ActionType.NAVIGATE_TO: ActionSchema(
        name=ActionType.NAVIGATE_TO.value,
        description="Navigate the robot to a target location",
        parameters={
            'x': ActionParameter('x', 'number', True, description="X coordinate"),
            'y': ActionParameter('y', 'number', True, description="Y coordinate"),
            'theta': ActionParameter('theta', 'number', False, 0.0,
                                    description="Target orientation (radians)"),
        },
        timeout=60.0,
        preconditions=["robot_localized", "path_clear"],
        effects=["robot_at_target"],
    ),

    ActionType.PICK: ActionSchema(
        name=ActionType.PICK.value,
        description="Pick up an object with the robot arm",
        parameters={
            'object_id': ActionParameter('object_id', 'string', True,
                                        description="ID of object to pick"),
            'grasp_type': ActionParameter('grasp_type', 'string', False, 'precision',
                                         constraints={'enum': ['precision', 'power']},
                                         description="Type of grasp to use"),
        },
        timeout=30.0,
        preconditions=["object_visible", "arm_ready", "gripper_open"],
        effects=["object_grasped"],
    ),

    ActionType.PLACE: ActionSchema(
        name=ActionType.PLACE.value,
        description="Place a held object at a location",
        parameters={
            'x': ActionParameter('x', 'number', True, description="X coordinate"),
            'y': ActionParameter('y', 'number', True, description="Y coordinate"),
            'z': ActionParameter('z', 'number', True, description="Z coordinate"),
        },
        timeout=30.0,
        preconditions=["object_grasped"],
        effects=["object_placed", "gripper_open"],
    ),

    ActionType.FIND_OBJECT: ActionSchema(
        name=ActionType.FIND_OBJECT.value,
        description="Search for and locate an object",
        parameters={
            'object_class': ActionParameter('object_class', 'string', True,
                                           description="Class/type of object"),
            'attributes': ActionParameter('attributes', 'object', False, {},
                                         description="Additional attributes (color, size)"),
        },
        timeout=45.0,
        preconditions=["camera_ready"],
        effects=["object_located"],
    ),

    ActionType.SPEAK: ActionSchema(
        name=ActionType.SPEAK.value,
        description="Speak a message using text-to-speech",
        parameters={
            'text': ActionParameter('text', 'string', True,
                                   description="Text to speak"),
            'language': ActionParameter('language', 'string', False, 'en',
                                       description="Language code"),
        },
        timeout=10.0,
        preconditions=[],
        effects=["message_spoken"],
    ),

    ActionType.WAIT: ActionSchema(
        name=ActionType.WAIT.value,
        description="Wait for a specified duration",
        parameters={
            'duration': ActionParameter('duration', 'number', True,
                                       constraints={'min': 0, 'max': 60},
                                       description="Duration in seconds"),
        },
        timeout=65.0,
        preconditions=[],
        effects=[],
    ),
}
