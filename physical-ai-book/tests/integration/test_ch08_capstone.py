#!/usr/bin/env python3
"""
Chapter 8: Capstone Integration Tests
Physical AI Book

Tests for the complete home assistant robot system including
coordinator, LLM planner, and safety monitor.

Usage:
    pytest tests/integration/test_ch08_capstone.py -v

Author: Physical AI Book
License: MIT
"""

import pytest
import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_ros():
    """Mock ROS 2 infrastructure."""
    with patch.dict('sys.modules', {
        'rclpy': MagicMock(),
        'rclpy.node': MagicMock(),
        'rclpy.action': MagicMock(),
        'rclpy.callback_groups': MagicMock(),
        'rclpy.executors': MagicMock(),
        'rclpy.qos': MagicMock(),
        'std_msgs': MagicMock(),
        'std_msgs.msg': MagicMock(),
        'geometry_msgs': MagicMock(),
        'geometry_msgs.msg': MagicMock(),
        'sensor_msgs': MagicMock(),
        'sensor_msgs.msg': MagicMock(),
        'nav2_msgs': MagicMock(),
        'nav2_msgs.action': MagicMock(),
    }):
        yield


@pytest.fixture
def llm_planner():
    """Create LLM planner instance."""
    from examples.capstone.llm_planner import LLMTaskPlanner
    return LLMTaskPlanner(use_llm=False)  # Use rule-based for tests


@pytest.fixture
def world_state():
    """Create test world state."""
    from examples.capstone.llm_planner import WorldState
    return WorldState(
        robot_location="home",
        robot_pose={'x': 0.0, 'y': 0.0, 'yaw': 0.0},
        known_objects={
            'cup': {'location': 'kitchen', 'position': [5.0, 2.0, 0.8]},
            'book': {'location': 'living room', 'position': [1.0, 0.5, 0.5]},
        },
        battery_level=80.0
    )


@pytest.fixture
def safety_config():
    """Create test safety configuration."""
    from examples.capstone.safety_monitor import SafetyConfig
    return SafetyConfig(
        min_obstacle_distance=0.3,
        slowdown_distance=0.8,
        max_linear_speed=0.5,
        max_angular_speed=1.0,
        battery_low=20.0,
        battery_critical=10.0,
        human_safety_radius=1.0,
    )


@pytest.fixture
def safety_checker(safety_config):
    """Create safety checker instance."""
    from examples.capstone.safety_monitor import SafetyChecker
    return SafetyChecker(config=safety_config)


# =============================================================================
# Coordinator Tests
# =============================================================================

class TestRobotState:
    """Test robot state enumeration."""

    def test_all_states_defined(self):
        """Test all expected states are defined."""
        from examples.capstone.coordinator import RobotState

        expected_states = [
            'IDLE', 'LISTENING', 'PLANNING', 'CONFIRMING',
            'EXECUTING', 'PAUSED', 'ERROR', 'EMERGENCY_STOP'
        ]

        for state_name in expected_states:
            assert hasattr(RobotState, state_name)

    def test_state_values_unique(self):
        """Test all state values are unique."""
        from examples.capstone.coordinator import RobotState

        values = [s.value for s in RobotState]
        assert len(values) == len(set(values))


class TestTaskPlan:
    """Test TaskPlan dataclass."""

    def test_task_plan_creation(self):
        """Test creating a task plan."""
        from examples.capstone.coordinator import TaskPlan

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
        assert plan.status == "pending"

    def test_current_action(self):
        """Test getting current action."""
        from examples.capstone.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test",
            actions=[{'type': 'a'}, {'type': 'b'}]
        )

        assert plan.current_action == {'type': 'a'}

    def test_advance_action(self):
        """Test advancing through actions."""
        from examples.capstone.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test",
            actions=[{'type': 'a'}, {'type': 'b'}]
        )

        plan.advance()
        assert plan.current_index == 1
        assert plan.current_action == {'type': 'b'}

        plan.advance()
        assert plan.is_complete

    def test_empty_plan_complete(self):
        """Test empty plan is immediately complete."""
        from examples.capstone.coordinator import TaskPlan

        plan = TaskPlan(goal="empty", actions=[])
        assert plan.is_complete
        assert plan.progress == 1.0

    def test_plan_progress(self):
        """Test plan progress calculation."""
        from examples.capstone.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test",
            actions=[{'type': 'a'}, {'type': 'b'}, {'type': 'c'}, {'type': 'd'}]
        )

        assert plan.progress == 0.0

        plan.advance()
        assert plan.progress == 0.25

        plan.advance()
        assert plan.progress == 0.5

    def test_plan_reset(self):
        """Test plan reset."""
        from examples.capstone.coordinator import TaskPlan

        plan = TaskPlan(
            goal="test",
            actions=[{'type': 'a'}, {'type': 'b'}]
        )

        plan.advance()
        plan.status = "executing"

        plan.reset()
        assert plan.current_index == 0
        assert plan.status == "pending"


# =============================================================================
# LLM Planner Tests
# =============================================================================

class TestLLMTaskPlanner:
    """Test LLM-based task planner."""

    def test_navigation_command(self, llm_planner):
        """Test navigation command parsing."""
        plan = llm_planner.plan("go to the kitchen")

        assert plan is not None
        assert 'kitchen' in plan['goal'].lower()
        assert any(a['type'] == 'navigate' for a in plan['actions'])

    def test_navigation_variations(self, llm_planner):
        """Test various navigation command formats."""
        commands = [
            "navigate to the bedroom",
            "move to the living room",
            "take me to the bathroom",
        ]

        for cmd in commands:
            plan = llm_planner.plan(cmd)
            assert plan is not None
            assert any(a['type'] == 'navigate' for a in plan['actions'])

    def test_fetch_command(self, llm_planner):
        """Test fetch command."""
        plan = llm_planner.plan("bring me a cup")

        assert plan is not None
        assert any(a['type'] == 'navigate' for a in plan['actions'])
        assert any(a.get('object') == 'cup' or 'cup' in a.get('message', '')
                   for a in plan['actions'])

    def test_fetch_unknown_object(self, llm_planner):
        """Test fetch with unrecognized object."""
        plan = llm_planner.plan("bring me something")

        assert plan is not None
        # Should ask for clarification
        assert any('speak' in a['type'] for a in plan['actions'])

    def test_pick_command(self, llm_planner):
        """Test pick up command."""
        plan = llm_planner.plan("pick up the ball")

        assert plan is not None
        assert any(a['type'] == 'pick' for a in plan['actions'])

    def test_stop_command(self, llm_planner):
        """Test stop command."""
        plan = llm_planner.plan("stop")

        assert plan is not None
        assert any(a['type'] == 'stop' for a in plan['actions'])

    def test_greeting_command(self, llm_planner):
        """Test greeting handling."""
        plan = llm_planner.plan("hello")

        assert plan is not None
        assert any(a['type'] == 'speak' for a in plan['actions'])

    def test_help_command(self, llm_planner):
        """Test help command."""
        plan = llm_planner.plan("what can you do")

        assert plan is not None
        assert any(a['type'] == 'speak' for a in plan['actions'])

    def test_unknown_command(self, llm_planner):
        """Test unknown command handling."""
        plan = llm_planner.plan("do something impossible xyz")

        assert plan is not None
        # Should respond with clarification
        assert any(a['type'] == 'speak' for a in plan['actions'])

    def test_world_state_integration(self, llm_planner, world_state):
        """Test planning with world state."""
        llm_planner.update_world_state(world_state)
        plan = llm_planner.plan("what is your status")

        assert plan is not None

    def test_recovery_plan_navigation(self, llm_planner):
        """Test recovery plan for navigation failure."""
        failed_action = {'type': 'navigate', 'location_name': 'kitchen'}
        recovery = llm_planner.get_recovery_plan(failed_action, "Path blocked")

        assert recovery is not None
        assert 'actions' in recovery

    def test_recovery_plan_pick(self, llm_planner):
        """Test recovery plan for pick failure."""
        failed_action = {'type': 'pick', 'object': 'cup'}
        recovery = llm_planner.get_recovery_plan(failed_action, "Grasp failed")

        assert recovery is not None
        assert any(a['type'] == 'speak' for a in recovery['actions'])


class TestWorldState:
    """Test WorldState dataclass."""

    def test_world_state_defaults(self):
        """Test default world state values."""
        from examples.capstone.llm_planner import WorldState

        state = WorldState()
        assert state.robot_location == "home"
        assert state.battery_level == 100.0
        assert state.held_object is None

    def test_world_state_to_context(self, world_state):
        """Test context string generation."""
        context = world_state.to_context()

        assert "home" in context
        assert "80" in context  # Battery level
        assert "cup" in context
        assert "kitchen" in context


class TestActionType:
    """Test ActionType enumeration."""

    def test_all_actions_defined(self):
        """Test all expected actions are defined."""
        from examples.capstone.llm_planner import ActionType

        expected = [
            'NAVIGATE', 'SPEAK', 'WAIT', 'STOP', 'PICK', 'PLACE',
            'FIND_OBJECT', 'NAVIGATE_TO_OBJECT', 'LOOK_AT',
            'OPEN_GRIPPER', 'CLOSE_GRIPPER'
        ]

        for action in expected:
            assert hasattr(ActionType, action)


# =============================================================================
# Safety Monitor Tests
# =============================================================================

class TestSafetyLevel:
    """Test SafetyLevel enumeration."""

    def test_safety_levels_ordered(self):
        """Test safety levels have increasing severity."""
        from examples.capstone.safety_monitor import SafetyLevel

        assert SafetyLevel.NORMAL.value < SafetyLevel.CAUTION.value
        assert SafetyLevel.CAUTION.value < SafetyLevel.WARNING.value
        assert SafetyLevel.WARNING.value < SafetyLevel.CRITICAL.value
        assert SafetyLevel.CRITICAL.value < SafetyLevel.EMERGENCY.value


class TestSafetyCategory:
    """Test SafetyCategory enumeration."""

    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        from examples.capstone.safety_monitor import SafetyCategory

        expected = [
            'COLLISION', 'BOUNDARY', 'BATTERY', 'THERMAL',
            'HUMAN_PROXIMITY', 'HARDWARE', 'SOFTWARE', 'WORKSPACE'
        ]

        for cat in expected:
            assert hasattr(SafetyCategory, cat)


class TestSafetyViolation:
    """Test SafetyViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a safety violation."""
        from examples.capstone.safety_monitor import (
            SafetyViolation, SafetyCategory, SafetyLevel
        )

        violation = SafetyViolation(
            category=SafetyCategory.COLLISION,
            level=SafetyLevel.WARNING,
            message="Obstacle detected",
            timestamp=time.time(),
            source="lidar",
            data={'distance': 0.5}
        )

        assert violation.category == SafetyCategory.COLLISION
        assert violation.level == SafetyLevel.WARNING

    def test_violation_to_dict(self):
        """Test violation serialization."""
        from examples.capstone.safety_monitor import (
            SafetyViolation, SafetyCategory, SafetyLevel
        )

        violation = SafetyViolation(
            category=SafetyCategory.BATTERY,
            level=SafetyLevel.CAUTION,
            message="Low battery",
            timestamp=1234567890.0
        )

        d = violation.to_dict()
        assert d['category'] == 'battery'
        assert d['level'] == 'CAUTION'
        assert d['message'] == "Low battery"


class TestSafetyConfig:
    """Test SafetyConfig dataclass."""

    def test_default_config(self):
        """Test default safety configuration."""
        from examples.capstone.safety_monitor import SafetyConfig

        config = SafetyConfig()

        assert config.min_obstacle_distance > 0
        assert config.max_linear_speed > 0
        assert config.battery_low > config.battery_critical

    def test_custom_config(self):
        """Test custom safety configuration."""
        from examples.capstone.safety_monitor import SafetyConfig

        config = SafetyConfig(
            min_obstacle_distance=0.5,
            max_linear_speed=1.0,
            battery_low=30.0
        )

        assert config.min_obstacle_distance == 0.5
        assert config.max_linear_speed == 1.0
        assert config.battery_low == 30.0


class TestSafetyChecker:
    """Test SafetyChecker class."""

    def test_check_navigation_target_valid(self, safety_checker):
        """Test valid navigation target."""
        target = {'x': 5.0, 'y': 3.0}
        current = {'x': 0.0, 'y': 0.0}

        is_safe, reason = safety_checker.check_navigation_target(target, current)
        assert is_safe

    def test_check_navigation_target_outside_workspace(self, safety_checker):
        """Test navigation target outside workspace."""
        target = {'x': 100.0, 'y': 0.0}  # Far outside
        current = {'x': 0.0, 'y': 0.0}

        is_safe, reason = safety_checker.check_navigation_target(target, current)
        assert not is_safe
        assert "outside" in reason.lower()

    def test_check_velocity_valid(self, safety_checker):
        """Test valid velocity."""
        is_safe, reason = safety_checker.check_velocity(0.3, 0.5)
        assert is_safe

    def test_check_velocity_linear_exceeded(self, safety_checker):
        """Test velocity exceeding linear limit."""
        is_safe, reason = safety_checker.check_velocity(1.0, 0.0)
        assert not is_safe
        assert "linear" in reason.lower()

    def test_check_velocity_angular_exceeded(self, safety_checker):
        """Test velocity exceeding angular limit."""
        is_safe, reason = safety_checker.check_velocity(0.0, 2.0)
        assert not is_safe
        assert "angular" in reason.lower()

    def test_check_manipulation_valid(self, safety_checker):
        """Test valid manipulation target."""
        position = [0.4, 0.0, 0.5]
        is_safe, reason = safety_checker.check_manipulation_target(position)
        assert is_safe

    def test_check_manipulation_too_far(self, safety_checker):
        """Test manipulation target too far."""
        position = [1.5, 0.0, 0.5]  # Beyond reach
        is_safe, reason = safety_checker.check_manipulation_target(position)
        assert not is_safe
        assert "far" in reason.lower()

    def test_check_manipulation_too_high(self, safety_checker):
        """Test manipulation target too high."""
        position = [0.3, 0.0, 2.0]  # Too high
        is_safe, reason = safety_checker.check_manipulation_target(position)
        assert not is_safe
        assert "high" in reason.lower()

    def test_check_manipulation_below_ground(self, safety_checker):
        """Test manipulation target below ground."""
        position = [0.3, 0.0, -0.1]  # Below ground
        is_safe, reason = safety_checker.check_manipulation_target(position)
        assert not is_safe


# =============================================================================
# Integration Tests
# =============================================================================

class TestCoordinatorIntegration:
    """Integration tests for coordinator with planner and safety."""

    def test_plan_with_safety_check(self, llm_planner, safety_checker):
        """Test planning and safety validation integration."""
        # Plan a navigation task
        plan = llm_planner.plan("go to the kitchen")

        # Check each navigation action
        for action in plan['actions']:
            if action['type'] == 'navigate':
                target = action.get('target', {})
                is_safe, _ = safety_checker.check_navigation_target(
                    target, {'x': 0, 'y': 0}
                )
                assert is_safe

    def test_fetch_plan_safety(self, llm_planner, safety_checker):
        """Test fetch plan with safety validation."""
        plan = llm_planner.plan("bring me a cup")

        # Verify all targets are within workspace
        for action in plan['actions']:
            if action['type'] == 'navigate':
                target = action.get('target', {})
                is_safe, _ = safety_checker.check_navigation_target(
                    target, {'x': 0, 'y': 0}
                )
                assert is_safe

    def test_recovery_triggered_on_violation(self, llm_planner, safety_checker):
        """Test recovery plan when safety violation detected."""
        # Simulate obstacle too close
        is_safe, reason = safety_checker.check_velocity(0.3, 0.0)

        # If moving too fast for conditions, get recovery
        failed_action = {'type': 'navigate', 'location_name': 'kitchen'}
        recovery = llm_planner.get_recovery_plan(failed_action, "Obstacle detected")

        assert recovery is not None


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_navigation_workflow(self, llm_planner, safety_checker, world_state):
        """Test complete navigation workflow."""
        # Set initial state
        llm_planner.update_world_state(world_state)

        # User command
        command = "go to the bedroom"
        plan = llm_planner.plan(command)

        assert plan is not None
        assert plan['goal']

        # Validate each action
        for i, action in enumerate(plan['actions']):
            action_type = action['type']

            if action_type == 'navigate':
                target = action['target']
                is_safe, _ = safety_checker.check_navigation_target(
                    target, world_state.robot_pose
                )
                assert is_safe, f"Action {i} has unsafe target"

            elif action_type == 'speak':
                assert 'message' in action

    def test_fetch_workflow(self, llm_planner, safety_checker, world_state):
        """Test complete fetch workflow."""
        llm_planner.update_world_state(world_state)

        plan = llm_planner.plan("fetch the cup from the kitchen")

        assert plan is not None

        # Should have multiple actions
        assert len(plan['actions']) >= 2

        # Should include navigation and pick actions
        action_types = [a['type'] for a in plan['actions']]
        assert 'navigate' in action_types
        # May include pick, find_object, etc.

    def test_emergency_stop_workflow(self, llm_planner, safety_checker):
        """Test emergency stop handling."""
        # Simulate critical situation
        is_safe, _ = safety_checker.check_velocity(2.0, 0.0)  # Too fast
        assert not is_safe

        # User issues stop command
        plan = llm_planner.plan("stop")
        assert any(a['type'] == 'stop' for a in plan['actions'])

    def test_low_battery_workflow(self, llm_planner, safety_checker):
        """Test low battery handling."""
        from examples.capstone.llm_planner import WorldState

        # Set low battery
        low_battery_state = WorldState(battery_level=15.0)
        llm_planner.update_world_state(low_battery_state)

        # Request status
        plan = llm_planner.plan("what is your status")

        assert plan is not None
        # Should mention battery in response
        assert any('speak' in a['type'] for a in plan['actions'])


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for planner."""

    def test_planning_speed(self, llm_planner):
        """Test planning completes quickly."""
        commands = [
            "go to the kitchen",
            "bring me a cup",
            "stop",
            "hello",
        ]

        for cmd in commands:
            start = time.time()
            plan = llm_planner.plan(cmd)
            elapsed = time.time() - start

            assert elapsed < 0.1, f"Planning '{cmd}' took {elapsed:.3f}s"
            assert plan is not None

    def test_multiple_plans_sequentially(self, llm_planner):
        """Test multiple planning requests."""
        commands = ["go to the " + loc for loc in [
            "kitchen", "bedroom", "bathroom", "living room"
        ]]

        start = time.time()
        for cmd in commands:
            plan = llm_planner.plan(cmd)
            assert plan is not None
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Multiple plans took {elapsed:.3f}s"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
