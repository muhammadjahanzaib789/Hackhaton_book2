#!/usr/bin/env python3
"""
LLM Response Validators
Physical AI Book - Chapter 6: LLM Integration

Validation middleware for LLM outputs to ensure type safety
and schema compliance before robot execution.

Features:
- JSON schema validation
- Action parameter validation
- Safety constraint checking
- Custom validation rules

Usage:
    from physical_ai_examples.llm.middleware import JSONValidator, ActionValidator

    # JSON validation
    validator = JSONValidator(schema)
    result = validator.validate(llm_output)

    # Action validation
    action_validator = ActionValidator(robot_constraints)
    is_safe, errors = action_validator.validate_action(action)

Author: Physical AI Book
License: MIT
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ..core.schemas import ActionSchema, RobotAction, ActionType, STANDARD_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of non-blocking warnings
        data: Validated/transformed data
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Optional[Any] = None

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(warning)


class JSONValidator:
    """
    Validates JSON output from LLMs against a schema.

    Handles common LLM output issues:
    - JSON embedded in markdown
    - Trailing commas
    - Unquoted strings
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.

        Args:
            schema: JSON schema for validation (optional)
        """
        self.schema = schema

    def validate(self, text: str) -> ValidationResult:
        """
        Validate and parse JSON from LLM output.

        Args:
            text: Raw LLM output text

        Returns:
            ValidationResult with parsed data
        """
        result = ValidationResult(is_valid=True)

        # Extract JSON from text
        json_str = self._extract_json(text)
        if not json_str:
            result.add_error("No valid JSON found in response")
            return result

        # Parse JSON
        try:
            data = json.loads(json_str)
            result.data = data
        except json.JSONDecodeError as e:
            result.add_error(f"JSON parse error: {e}")
            return result

        # Validate against schema
        if self.schema:
            schema_errors = self._validate_schema(data, self.schema)
            for error in schema_errors:
                result.add_error(error)

        return result

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from LLM output.

        Handles:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON mixed with text
        """
        # Check for markdown code block
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        match = re.search(code_block_pattern, text)
        if match:
            return match.group(1).strip()

        # Try to find JSON object or array
        # Find first { or [ and last } or ]
        json_start = -1
        json_end = -1

        for i, char in enumerate(text):
            if char in '{[' and json_start == -1:
                json_start = i
                break

        for i in range(len(text) - 1, -1, -1):
            if text[i] in '}]':
                json_end = i + 1
                break

        if json_start >= 0 and json_end > json_start:
            return text[json_start:json_end]

        return None

    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = ""
    ) -> List[str]:
        """
        Validate data against JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema
            path: Current path in data structure

        Returns:
            List of validation errors
        """
        errors = []
        schema_type = schema.get('type')

        # Type validation
        if schema_type:
            if not self._check_type(data, schema_type):
                errors.append(
                    f"At '{path}': expected {schema_type}, got {type(data).__name__}"
                )
                return errors

        # Object validation
        if schema_type == 'object':
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            # Check required fields
            for field in required:
                if field not in data:
                    errors.append(f"At '{path}': missing required field '{field}'")

            # Validate properties
            for key, value in data.items():
                if key in properties:
                    field_path = f"{path}.{key}" if path else key
                    errors.extend(
                        self._validate_schema(value, properties[key], field_path)
                    )

        # Array validation
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"
                errors.extend(
                    self._validate_schema(item, items_schema, item_path)
                )

        # Numeric constraints
        if schema_type in ('number', 'integer'):
            if 'minimum' in schema and data < schema['minimum']:
                errors.append(f"At '{path}': {data} below minimum {schema['minimum']}")
            if 'maximum' in schema and data > schema['maximum']:
                errors.append(f"At '{path}': {data} above maximum {schema['maximum']}")

        # String constraints
        if schema_type == 'string':
            if 'enum' in schema and data not in schema['enum']:
                errors.append(f"At '{path}': '{data}' not in enum {schema['enum']}")
            if 'pattern' in schema:
                if not re.match(schema['pattern'], data):
                    errors.append(f"At '{path}': doesn't match pattern")

        return errors

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }
        return isinstance(value, type_map.get(expected, object))


class ActionValidator:
    """
    Validates robot actions for safety and executability.

    Checks:
    - Action type validity
    - Parameter constraints
    - Safety limits
    - Physical feasibility
    """

    # Default safety constraints
    DEFAULT_CONSTRAINTS = {
        'max_speed': 1.0,           # m/s
        'max_acceleration': 2.0,     # m/s^2
        'max_force': 50.0,           # N
        'max_reach': 1.0,            # m
        'min_clearance': 0.05,       # m
        'restricted_zones': [],      # List of (x, y, radius) tuples
        'restricted_objects': [      # Objects that cannot be manipulated
            'human', 'person', 'knife', 'hot', 'sharp'
        ],
    }

    def __init__(
        self,
        constraints: Optional[Dict[str, Any]] = None,
        custom_validators: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize action validator.

        Args:
            constraints: Safety constraints
            custom_validators: Custom validation functions per action type
        """
        self.constraints = {**self.DEFAULT_CONSTRAINTS, **(constraints or {})}
        self.custom_validators = custom_validators or {}
        self.schemas = STANDARD_SCHEMAS

    def validate_action(self, action: RobotAction) -> ValidationResult:
        """
        Validate a single robot action.

        Args:
            action: Action to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check action type
        if action.action_type not in ActionType:
            result.add_error(f"Unknown action type: {action.action_type}")
            return result

        # Get schema
        schema = self.schemas.get(action.action_type)
        if schema:
            try:
                schema.validate_parameters(action.parameters)
            except ValueError as e:
                result.add_error(str(e))

        # Check safety constraints
        self._check_safety(action, result)

        # Run custom validators
        if action.action_type.value in self.custom_validators:
            validator = self.custom_validators[action.action_type.value]
            try:
                validator(action, result)
            except Exception as e:
                result.add_error(f"Custom validation error: {e}")

        return result

    def validate_plan(
        self,
        actions: List[RobotAction]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate a sequence of actions.

        Args:
            actions: List of actions to validate

        Returns:
            (overall_valid, list of results)
        """
        results = []
        overall_valid = True

        for action in actions:
            result = self.validate_action(action)
            results.append(result)
            if not result.is_valid:
                overall_valid = False

        return overall_valid, results

    def _check_safety(
        self,
        action: RobotAction,
        result: ValidationResult
    ) -> None:
        """Check action against safety constraints."""

        params = action.parameters

        # Speed check
        if 'speed' in params:
            if params['speed'] > self.constraints['max_speed']:
                result.add_error(
                    f"Speed {params['speed']} exceeds max {self.constraints['max_speed']}"
                )

        # Force check
        if 'force' in params:
            if params['force'] > self.constraints['max_force']:
                result.add_error(
                    f"Force {params['force']} exceeds max {self.constraints['max_force']}"
                )

        # Reach check for position targets
        if 'x' in params and 'y' in params:
            distance = (params['x']**2 + params['y']**2) ** 0.5
            if distance > self.constraints['max_reach']:
                result.add_warning(
                    f"Target distance {distance:.2f}m may be beyond reach"
                )

        # Restricted zone check
        if 'x' in params and 'y' in params:
            for zone in self.constraints['restricted_zones']:
                zx, zy, radius = zone
                dist = ((params['x'] - zx)**2 + (params['y'] - zy)**2) ** 0.5
                if dist < radius:
                    result.add_error(
                        f"Target ({params['x']}, {params['y']}) is in restricted zone"
                    )

        # Object restriction check
        if 'object_id' in params:
            obj = str(params['object_id']).lower()
            for restricted in self.constraints['restricted_objects']:
                if restricted in obj:
                    result.add_error(
                        f"Cannot manipulate restricted object: {params['object_id']}"
                    )


class SafetyFilter:
    """
    High-level safety filter for LLM outputs.

    Combines JSON validation and action validation with
    additional safety checks.
    """

    def __init__(
        self,
        action_validator: Optional[ActionValidator] = None,
        blocked_phrases: Optional[List[str]] = None
    ):
        """
        Initialize safety filter.

        Args:
            action_validator: Action validator instance
            blocked_phrases: Phrases that should trigger rejection
        """
        self.action_validator = action_validator or ActionValidator()
        self.blocked_phrases = blocked_phrases or [
            'ignore safety',
            'bypass',
            'override',
            'harm',
            'damage',
        ]

    def filter_response(
        self,
        response: str,
        actions: Optional[List[RobotAction]] = None
    ) -> ValidationResult:
        """
        Filter LLM response for safety.

        Args:
            response: Raw LLM response
            actions: Parsed actions (optional)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Check for blocked phrases
        response_lower = response.lower()
        for phrase in self.blocked_phrases:
            if phrase in response_lower:
                result.add_error(f"Response contains blocked phrase: '{phrase}'")

        # Validate actions if provided
        if actions:
            overall_valid, action_results = self.action_validator.validate_plan(
                actions
            )
            if not overall_valid:
                for i, ar in enumerate(action_results):
                    for error in ar.errors:
                        result.add_error(f"Action {i}: {error}")

        return result
