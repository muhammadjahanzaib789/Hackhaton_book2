#!/usr/bin/env python3
"""
Integration Tests for Chapter 6: LLM Integration
Physical AI Book

Tests the LLM integration pipeline including:
- Provider abstraction
- Action schema validation
- Task planning
- Safety validation
- Rate limiting

These tests use mock LLM responses to enable testing without
actual LLM API access.

Usage:
    pytest tests/integration/test_ch06_llm_integration.py -v

Author: Physical AI Book
License: MIT
"""

import asyncio
import json
import pytest
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/examples'))

from llm.core.provider import LLMConfig, LLMProvider, LLMResponse
from llm.core.schemas import (
    ActionType,
    ActionParameter,
    ActionSchema,
    RobotAction,
    ActionResponse,
    STANDARD_SCHEMAS
)
from llm.middleware.validator import JSONValidator, ActionValidator, ValidationResult
from llm.middleware.rate_limiter import RateLimiter, TokenBucket
from llm.robotics.action_planner import ActionPlanner, WorldState
from llm.robotics.safety_validator import SafetyValidator, RobotConstraints, SafetyLevel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider='mock',
        model='test-model',
        temperature=0.0,
        max_tokens=1024
    )


@pytest.fixture
def mock_llm_provider(llm_config):
    """Create a mock LLM provider for testing."""

    class MockProvider(LLMProvider):
        """Mock LLM provider for testing."""

        def __init__(self, config: LLMConfig):
            super().__init__(config)
            self.responses = []
            self.call_count = 0

        def set_response(self, response: str):
            """Set the next response to return."""
            self.responses.append(response)

        async def generate(
            self,
            prompt: str,
            *,
            system_prompt: Optional[str] = None,
            json_mode: bool = False,
            **kwargs
        ) -> LLMResponse:
            self.call_count += 1

            if self.responses:
                content = self.responses.pop(0)
            else:
                content = '{"message": "default response"}'

            return LLMResponse(
                content=content,
                model=self.config.model,
                usage={'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30},
                finish_reason='stop'
            )

        async def generate_stream(
            self,
            prompt: str,
            *,
            system_prompt: Optional[str] = None,
            **kwargs
        ) -> AsyncGenerator[str, None]:
            content = "streaming response"
            for word in content.split():
                yield word + " "

        async def health_check(self) -> bool:
            return True

    # Register the mock provider
    LLMProvider._providers['mock'] = MockProvider

    return MockProvider(llm_config)


@pytest.fixture
def robot_constraints():
    """Create test robot constraints."""
    return RobotConstraints(
        max_speed=0.5,
        max_force=20.0,
        max_reach=0.8,
        workspace_bounds=((-2.0, -2.0), (2.0, 2.0)),
        restricted_zones=[(1.5, 1.5, 0.3)],
        min_human_distance=0.5,
        max_payload=1.5
    )


@pytest.fixture
def world_state():
    """Create a test world state."""
    return WorldState(
        robot_position=(0.0, 0.0, 0.0),
        held_object=None,
        visible_objects=[
            {'id': 'red_cup', 'type': 'cup', 'color': 'red', 'position': (0.5, 0.3, 0.1)},
            {'id': 'blue_ball', 'type': 'ball', 'color': 'blue', 'position': (0.7, 0.1, 0.05)},
        ],
        known_locations={
            'table': (1.0, 0.0),
            'shelf': (0.0, 1.5),
            'home': (0.0, 0.0),
        },
        battery_level=85.0,
        arm_state='ready'
    )


# =============================================================================
# LLM Config Tests
# =============================================================================

class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == 'ollama'
        assert config.model == 'llama3.2'
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            provider='openai',
            model='gpt-4',
            temperature=0.5,
            max_tokens=2048,
            api_key='test-key'
        )
        assert config.provider == 'openai'
        assert config.model == 'gpt-4'
        assert config.api_key == 'test-key'

    def test_temperature_validation(self):
        """Test temperature range validation."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.5)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.5)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


# =============================================================================
# LLM Provider Tests
# =============================================================================

class TestLLMProvider:
    """Test LLM provider functionality."""

    @pytest.mark.asyncio
    async def test_provider_generate(self, mock_llm_provider):
        """Test basic generation."""
        mock_llm_provider.set_response('{"result": "test output"}')

        response = await mock_llm_provider.generate("Test prompt")

        assert response.content == '{"result": "test output"}'
        assert response.model == 'test-model'
        assert response.total_tokens == 30

    @pytest.mark.asyncio
    async def test_provider_health_check(self, mock_llm_provider):
        """Test health check."""
        is_healthy = await mock_llm_provider.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_provider_generate_json(self, mock_llm_provider):
        """Test JSON generation and parsing."""
        mock_llm_provider.set_response('{"key": "value", "number": 42}')

        result = await mock_llm_provider.generate_json("Generate JSON")

        assert result['key'] == 'value'
        assert result['number'] == 42


# =============================================================================
# Action Schema Tests
# =============================================================================

class TestActionSchemas:
    """Test action schema validation."""

    def test_action_parameter_validation(self):
        """Test parameter validation."""
        param = ActionParameter(
            name='speed',
            param_type='number',
            required=True,
            constraints={'min': 0.0, 'max': 1.0}
        )

        # Valid value
        assert param.validate(0.5) is True

        # Below minimum
        with pytest.raises(ValueError):
            param.validate(-0.5)

        # Above maximum
        with pytest.raises(ValueError):
            param.validate(1.5)

    def test_action_schema_validate_params(self):
        """Test action schema parameter validation."""
        schema = ActionSchema(
            name='test_action',
            parameters={
                'target': ActionParameter('target', 'string', required=True),
                'speed': ActionParameter('speed', 'number', required=False, default=0.5),
            }
        )

        # Valid params
        assert schema.validate_parameters({'target': 'location1'}) is True
        assert schema.validate_parameters({'target': 'loc', 'speed': 0.3}) is True

        # Missing required param
        with pytest.raises(ValueError):
            schema.validate_parameters({'speed': 0.5})

        # Unknown param
        with pytest.raises(ValueError):
            schema.validate_parameters({'target': 'loc', 'unknown': 'value'})

    def test_robot_action_creation(self):
        """Test robot action creation."""
        action = RobotAction(
            action_type=ActionType.NAVIGATE_TO,
            parameters={'x': 1.0, 'y': 2.0, 'theta': 0.0}
        )

        assert action.action_type == ActionType.NAVIGATE_TO
        assert action.parameters['x'] == 1.0
        assert action.parameters['y'] == 2.0

    def test_robot_action_from_dict(self):
        """Test creating robot action from dictionary."""
        data = {
            'action_type': 'pick',
            'parameters': {'object_id': 'cup_1'},
            'timeout': 30.0
        }

        action = RobotAction.from_dict(data)

        assert action.action_type == ActionType.PICK
        assert action.parameters['object_id'] == 'cup_1'
        assert action.timeout == 30.0

    def test_action_response_parsing(self):
        """Test parsing action response."""
        llm_output = {
            'goal': 'Pick up the red cup',
            'plan': [
                {'action_type': 'find_object', 'parameters': {'object_class': 'cup'}},
                {'action_type': 'navigate_to', 'parameters': {'x': 0.5, 'y': 0.3, 'theta': 0}},
                {'action_type': 'pick', 'parameters': {'object_id': 'red_cup'}},
            ],
            'reasoning': 'Locate, approach, then pick',
            'confidence': 0.85
        }

        response = ActionResponse.from_llm_output(llm_output)

        assert response.goal == 'Pick up the red cup'
        assert len(response.plan) == 3
        assert response.plan[0].action_type == ActionType.FIND_OBJECT
        assert response.confidence == 0.85


# =============================================================================
# JSON Validator Tests
# =============================================================================

class TestJSONValidator:
    """Test JSON validation middleware."""

    def test_validate_pure_json(self):
        """Test validation of pure JSON."""
        validator = JSONValidator()

        result = validator.validate('{"key": "value"}')

        assert result.is_valid is True
        assert result.data == {'key': 'value'}

    def test_validate_json_in_markdown(self):
        """Test extraction of JSON from markdown."""
        validator = JSONValidator()

        text = '''Here's the response:
```json
{"result": "success", "value": 42}
```
'''
        result = validator.validate(text)

        assert result.is_valid is True
        assert result.data['result'] == 'success'
        assert result.data['value'] == 42

    def test_validate_json_with_text(self):
        """Test extraction of JSON mixed with text."""
        validator = JSONValidator()

        text = 'The answer is {"status": "ok", "count": 5} as expected.'
        result = validator.validate(text)

        assert result.is_valid is True
        assert result.data['status'] == 'ok'

    def test_validate_against_schema(self):
        """Test validation against JSON schema."""
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'count': {'type': 'integer', 'minimum': 0}
            },
            'required': ['name']
        }
        validator = JSONValidator(schema)

        # Valid
        result = validator.validate('{"name": "test", "count": 5}')
        assert result.is_valid is True

        # Missing required field
        result = validator.validate('{"count": 5}')
        assert result.is_valid is False

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        validator = JSONValidator()

        result = validator.validate('this is not json')
        assert result.is_valid is False
        assert len(result.errors) > 0


# =============================================================================
# Action Validator Tests
# =============================================================================

class TestActionValidator:
    """Test action validation middleware."""

    def test_validate_valid_action(self, robot_constraints):
        """Test validation of a valid action."""
        validator = ActionValidator(constraints={
            'max_speed': robot_constraints.max_speed,
            'max_force': robot_constraints.max_force
        })

        action = RobotAction(
            action_type=ActionType.NAVIGATE_TO,
            parameters={'x': 1.0, 'y': 1.0, 'speed': 0.3}
        )

        result = validator.validate_action(action)

        assert result.is_valid is True

    def test_validate_speed_violation(self, robot_constraints):
        """Test detection of speed violation."""
        validator = ActionValidator(constraints={
            'max_speed': 0.5
        })

        action = RobotAction(
            action_type=ActionType.NAVIGATE_TO,
            parameters={'x': 1.0, 'y': 1.0, 'speed': 1.0}
        )

        result = validator.validate_action(action)

        assert result.is_valid is False
        assert any('speed' in e.lower() for e in result.errors)

    def test_validate_restricted_object(self):
        """Test detection of restricted object manipulation."""
        validator = ActionValidator()

        action = RobotAction(
            action_type=ActionType.PICK,
            parameters={'object_id': 'sharp_knife'}
        )

        result = validator.validate_action(action)

        assert result.is_valid is False

    def test_validate_plan(self):
        """Test validation of action sequence."""
        validator = ActionValidator()

        actions = [
            RobotAction(ActionType.FIND_OBJECT, {'object_class': 'cup'}),
            RobotAction(ActionType.NAVIGATE_TO, {'x': 1.0, 'y': 0.5, 'theta': 0}),
            RobotAction(ActionType.PICK, {'object_id': 'cup_1'}),
        ]

        is_valid, results = validator.validate_plan(actions)

        assert is_valid is True


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_token_bucket_basic(self):
        """Test basic token bucket operation."""
        bucket = TokenBucket(rate=10.0, capacity=20.0)

        # Should start full
        assert bucket.try_acquire(1.0) is True

        # Should be able to acquire up to capacity
        for _ in range(19):
            assert bucket.try_acquire(1.0) is True

        # Should fail when empty
        assert bucket.try_acquire(1.0) is False

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test rate limiter acquire."""
        limiter = RateLimiter(rate=100.0, burst=10.0)

        # Should allow burst
        for _ in range(10):
            await limiter.acquire()

        assert limiter.stats['requests'] == 10

    def test_rate_limiter_try_acquire(self):
        """Test non-blocking acquire."""
        limiter = RateLimiter(rate=10.0, burst=5.0)

        # Should succeed up to burst
        for _ in range(5):
            assert limiter.try_acquire() is True

        # Should fail when exhausted
        assert limiter.try_acquire() is False


# =============================================================================
# Safety Validator Tests
# =============================================================================

class TestSafetyValidator:
    """Test safety validation functionality."""

    def test_workspace_boundary_check(self, robot_constraints):
        """Test workspace boundary enforcement."""
        validator = SafetyValidator(robot_constraints)

        # Valid position
        action = RobotAction(
            ActionType.NAVIGATE_TO,
            {'x': 1.0, 'y': 1.0}
        )
        is_safe, violations = validator.validate_action(action)
        assert is_safe is True

        # Outside workspace
        action = RobotAction(
            ActionType.NAVIGATE_TO,
            {'x': 10.0, 'y': 10.0}
        )
        is_safe, violations = validator.validate_action(action)
        assert is_safe is False

    def test_restricted_zone_check(self, robot_constraints):
        """Test restricted zone enforcement."""
        validator = SafetyValidator(robot_constraints)

        # Position in restricted zone (1.5, 1.5, radius 0.3)
        action = RobotAction(
            ActionType.NAVIGATE_TO,
            {'x': 1.5, 'y': 1.5}
        )
        is_safe, violations = validator.validate_action(action)

        assert is_safe is False
        assert any(v.level == SafetyLevel.CRITICAL for v in violations)

    def test_human_proximity_check(self, robot_constraints):
        """Test human proximity safety."""
        validator = SafetyValidator(robot_constraints)
        validator.update_human_positions([(1.0, 1.0)])

        # Too close to human
        action = RobotAction(
            ActionType.NAVIGATE_TO,
            {'x': 1.1, 'y': 1.0}
        )
        is_safe, violations = validator.validate_action(action)

        assert is_safe is False

    def test_forbidden_object_check(self, robot_constraints):
        """Test forbidden object manipulation check."""
        validator = SafetyValidator(robot_constraints)

        action = RobotAction(
            ActionType.PICK,
            {'object_id': 'hot_coffee'}
        )
        is_safe, violations = validator.validate_action(action)

        assert is_safe is False
        assert any('forbidden' in str(v).lower() for v in violations)

    def test_emergency_stop_check(self, robot_constraints):
        """Test emergency stop triggering."""
        validator = SafetyValidator(robot_constraints)

        action = RobotAction(ActionType.NAVIGATE_TO, {'x': 0.5, 'y': 0.5})

        # Should not stop normally
        current_state = {'nearest_human_distance': 2.0}
        should_stop = validator.emergency_stop_check(action, current_state)
        assert should_stop is False

        # Should stop on human proximity
        current_state = {'nearest_human_distance': 0.2}
        should_stop = validator.emergency_stop_check(action, current_state)
        assert should_stop is True

        # Should stop on contact
        current_state = {'contact_detected': True}
        should_stop = validator.emergency_stop_check(action, current_state)
        assert should_stop is True


# =============================================================================
# Action Planner Tests
# =============================================================================

class TestActionPlanner:
    """Test action planning functionality."""

    @pytest.mark.asyncio
    async def test_plan_simple_command(self, mock_llm_provider, world_state):
        """Test planning a simple command."""
        # Set mock response
        mock_response = json.dumps({
            'goal': 'Pick up the red cup',
            'plan': [
                {'action_type': 'find_object', 'parameters': {'object_class': 'cup', 'attributes': {'color': 'red'}}},
                {'action_type': 'pick', 'parameters': {'object_id': 'red_cup'}},
            ],
            'reasoning': 'Locate the cup then pick it up',
            'confidence': 0.9
        })
        mock_llm_provider.set_response(mock_response)

        planner = ActionPlanner(mock_llm_provider)
        plan = await planner.plan("Pick up the red cup", world_state)

        assert plan.goal == 'Pick up the red cup'
        assert len(plan.plan) == 2
        assert plan.plan[0].action_type == ActionType.FIND_OBJECT
        assert plan.plan[1].action_type == ActionType.PICK
        assert plan.confidence == 0.9

    @pytest.mark.asyncio
    async def test_plan_with_low_confidence(self, mock_llm_provider, world_state):
        """Test handling of low confidence plans."""
        mock_response = json.dumps({
            'goal': 'Fly to the moon',
            'plan': [],
            'reasoning': 'Robot cannot fly',
            'confidence': 0.1
        })
        mock_llm_provider.set_response(mock_response)

        planner = ActionPlanner(mock_llm_provider, min_confidence=0.3)
        plan = await planner.plan("Fly to the moon", world_state)

        assert plan.confidence < 0.3
        assert len(plan.plan) == 0

    @pytest.mark.asyncio
    async def test_task_decomposition(self, mock_llm_provider):
        """Test task decomposition."""
        mock_response = json.dumps({
            'subtasks': [
                'Navigate to kitchen',
                'Find the coffee machine',
                'Place cup under dispenser',
                'Press brew button',
                'Wait for brewing',
                'Pick up cup',
                'Return to user'
            ]
        })
        mock_llm_provider.set_response(mock_response)

        planner = ActionPlanner(mock_llm_provider)
        subtasks = await planner.decompose_task("Make me a coffee")

        assert len(subtasks) == 7
        assert 'Navigate to kitchen' in subtasks

    @pytest.mark.asyncio
    async def test_recovery_planning(self, mock_llm_provider, world_state):
        """Test recovery planning after failure."""
        mock_response = json.dumps({
            'goal': 'Recovery: Pick up the red cup',
            'plan': [
                {'action_type': 'scan_area', 'parameters': {}},
                {'action_type': 'find_object', 'parameters': {'object_class': 'cup'}},
                {'action_type': 'navigate_to', 'parameters': {'x': 0.6, 'y': 0.3, 'theta': 0}},
                {'action_type': 'pick', 'parameters': {'object_id': 'red_cup'}},
            ],
            'reasoning': 'Rescan area and try different approach',
            'confidence': 0.7
        })
        mock_llm_provider.set_response(mock_response)

        planner = ActionPlanner(mock_llm_provider)

        original_plan = ActionResponse(
            goal='Pick up the red cup',
            plan=[RobotAction(ActionType.PICK, {'object_id': 'red_cup'})],
            reasoning='Direct pick',
            confidence=0.9
        )
        failed_action = original_plan.plan[0]

        recovery = await planner.replan_on_failure(
            original_plan,
            failed_action,
            "Object not reachable",
            world_state
        )

        assert 'Recovery' in recovery.goal
        assert len(recovery.plan) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMRoboticsIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_llm_provider, robot_constraints, world_state):
        """Test complete LLM → Planning → Validation pipeline."""
        # 1. Set up mock LLM response
        mock_response = json.dumps({
            'goal': 'Fetch the red cup',
            'plan': [
                {'action_type': 'find_object', 'parameters': {'object_class': 'cup', 'attributes': {'color': 'red'}}},
                {'action_type': 'navigate_to', 'parameters': {'x': 0.5, 'y': 0.3, 'theta': 0}},
                {'action_type': 'pick', 'parameters': {'object_id': 'red_cup'}},
            ],
            'reasoning': 'Locate, approach, and pick up the cup',
            'confidence': 0.85
        })
        mock_llm_provider.set_response(mock_response)

        # 2. Plan action
        planner = ActionPlanner(mock_llm_provider)
        plan = await planner.plan("Bring me the red cup", world_state)

        assert len(plan.plan) == 3

        # 3. Validate plan
        validator = SafetyValidator(robot_constraints)
        is_safe, violations = validator.validate_plan(plan.plan)

        assert is_safe is True

        # 4. Verify action sequence makes sense
        action_types = [a.action_type for a in plan.plan]
        assert ActionType.FIND_OBJECT in action_types
        assert ActionType.PICK in action_types


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
