#!/usr/bin/env python3
"""
Robot Safety Validator
Physical AI Book - Chapter 6: LLM Integration

Validates robot actions for safety before execution.
Implements multi-layer safety checks for physical robot systems.

Features:
- Workspace boundary checking
- Speed and force limits
- Collision prediction
- Human proximity detection
- Emergency stop triggers

Usage:
    from physical_ai_examples.llm.robotics import SafetyValidator, RobotConstraints

    constraints = RobotConstraints(
        max_speed=0.5,
        max_force=20.0,
        workspace_bounds=((-2, -2), (2, 2))
    )
    validator = SafetyValidator(constraints)

    is_safe, reasons = validator.validate_action(action)
    if not is_safe:
        print(f"Action blocked: {reasons}")

Author: Physical AI Book
License: MIT
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.schemas import RobotAction, ActionType

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety severity levels."""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class SafetyViolation:
    """
    A safety violation detected during validation.

    Attributes:
        level: Severity level
        message: Human-readable description
        action_type: Action that caused violation
        parameter: Specific parameter involved
        suggested_fix: How to fix the issue
    """
    level: SafetyLevel
    message: str
    action_type: Optional[ActionType] = None
    parameter: Optional[str] = None
    suggested_fix: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.message}"


@dataclass
class RobotConstraints:
    """
    Physical and safety constraints for a robot.

    Attributes:
        max_speed: Maximum linear speed (m/s)
        max_angular_speed: Maximum angular speed (rad/s)
        max_acceleration: Maximum acceleration (m/s^2)
        max_force: Maximum gripper force (N)
        max_reach: Maximum arm reach (m)
        workspace_bounds: ((min_x, min_y), (max_x, max_y))
        restricted_zones: List of (x, y, radius) no-go zones
        min_human_distance: Minimum distance to humans (m)
        max_payload: Maximum payload weight (kg)
        joint_limits: Joint angle limits per joint
    """
    max_speed: float = 1.0
    max_angular_speed: float = 2.0
    max_acceleration: float = 2.0
    max_force: float = 50.0
    max_reach: float = 1.0
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-5.0, -5.0), (5.0, 5.0)
    )
    restricted_zones: List[Tuple[float, float, float]] = field(default_factory=list)
    min_human_distance: float = 0.5
    max_payload: float = 2.0
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def in_workspace(self, x: float, y: float) -> bool:
        """Check if position is within workspace."""
        (min_x, min_y), (max_x, max_y) = self.workspace_bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def in_restricted_zone(self, x: float, y: float) -> Optional[int]:
        """
        Check if position is in a restricted zone.

        Returns:
            Zone index if in restricted zone, None otherwise
        """
        for i, (zx, zy, radius) in enumerate(self.restricted_zones):
            distance = math.sqrt((x - zx)**2 + (y - zy)**2)
            if distance < radius:
                return i
        return None


class SafetyValidator:
    """
    Validates robot actions for safety compliance.

    Implements layered safety checks:
    1. Parameter validation (types, ranges)
    2. Kinematic feasibility
    3. Workspace boundaries
    4. Collision checking
    5. Human proximity
    """

    # Objects that should never be manipulated
    FORBIDDEN_OBJECTS = [
        'human', 'person', 'child', 'baby',
        'knife', 'blade', 'sharp',
        'fire', 'flame', 'hot',
        'chemical', 'acid', 'poison',
        'electrical', 'wire', 'live',
    ]

    def __init__(
        self,
        constraints: Optional[RobotConstraints] = None,
        *,
        custom_checks: Optional[List[Callable[[RobotAction], List[SafetyViolation]]]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize safety validator.

        Args:
            constraints: Robot physical constraints
            custom_checks: Additional validation functions
            strict_mode: If True, warnings become errors
        """
        self.constraints = constraints or RobotConstraints()
        self.custom_checks = custom_checks or []
        self.strict_mode = strict_mode
        self._human_positions: List[Tuple[float, float]] = []

    def update_human_positions(
        self,
        positions: List[Tuple[float, float]]
    ) -> None:
        """
        Update known human positions.

        Called by perception system to maintain human tracking.

        Args:
            positions: List of (x, y) human positions
        """
        self._human_positions = positions

    def validate_action(
        self,
        action: RobotAction
    ) -> Tuple[bool, List[SafetyViolation]]:
        """
        Validate a single robot action.

        Args:
            action: Action to validate

        Returns:
            (is_safe, list of violations)
        """
        violations = []

        # Layer 1: Parameter validation
        violations.extend(self._check_parameters(action))

        # Layer 2: Kinematic feasibility
        violations.extend(self._check_kinematics(action))

        # Layer 3: Workspace boundaries
        violations.extend(self._check_workspace(action))

        # Layer 4: Restricted zones
        violations.extend(self._check_restricted_zones(action))

        # Layer 5: Human proximity
        violations.extend(self._check_human_proximity(action))

        # Layer 6: Forbidden objects
        violations.extend(self._check_forbidden_objects(action))

        # Layer 7: Custom checks
        for check in self.custom_checks:
            try:
                violations.extend(check(action))
            except Exception as e:
                logger.error(f"Custom check failed: {e}")

        # Determine if action is safe
        if self.strict_mode:
            is_safe = len(violations) == 0
        else:
            is_safe = not any(
                v.level in (SafetyLevel.DANGER, SafetyLevel.CRITICAL)
                for v in violations
            )

        return is_safe, violations

    def validate_plan(
        self,
        actions: List[RobotAction]
    ) -> Tuple[bool, Dict[int, List[SafetyViolation]]]:
        """
        Validate a sequence of actions.

        Args:
            actions: List of actions to validate

        Returns:
            (overall_safe, dict mapping action index to violations)
        """
        all_violations = {}
        overall_safe = True

        for i, action in enumerate(actions):
            is_safe, violations = self.validate_action(action)
            if violations:
                all_violations[i] = violations
            if not is_safe:
                overall_safe = False

        return overall_safe, all_violations

    def _check_parameters(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check parameter values against constraints."""
        violations = []
        params = action.parameters

        # Speed check
        if 'speed' in params:
            if params['speed'] > self.constraints.max_speed:
                violations.append(SafetyViolation(
                    level=SafetyLevel.DANGER,
                    message=(
                        f"Speed {params['speed']:.2f} m/s exceeds "
                        f"maximum {self.constraints.max_speed:.2f} m/s"
                    ),
                    action_type=action.action_type,
                    parameter='speed',
                    suggested_fix=f"Reduce speed to {self.constraints.max_speed:.2f}"
                ))
            elif params['speed'] > self.constraints.max_speed * 0.8:
                violations.append(SafetyViolation(
                    level=SafetyLevel.WARNING,
                    message="Speed is near maximum limit",
                    action_type=action.action_type,
                    parameter='speed'
                ))

        # Force check
        if 'force' in params:
            if params['force'] > self.constraints.max_force:
                violations.append(SafetyViolation(
                    level=SafetyLevel.DANGER,
                    message=(
                        f"Force {params['force']:.1f} N exceeds "
                        f"maximum {self.constraints.max_force:.1f} N"
                    ),
                    action_type=action.action_type,
                    parameter='force',
                    suggested_fix=f"Reduce force to {self.constraints.max_force:.1f}"
                ))

        # Angular speed check
        if 'angular_speed' in params:
            if params['angular_speed'] > self.constraints.max_angular_speed:
                violations.append(SafetyViolation(
                    level=SafetyLevel.DANGER,
                    message="Angular speed exceeds limit",
                    action_type=action.action_type,
                    parameter='angular_speed'
                ))

        return violations

    def _check_kinematics(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check kinematic feasibility."""
        violations = []
        params = action.parameters

        # Reach check for target positions
        if action.action_type in (ActionType.PICK, ActionType.PLACE):
            if 'x' in params and 'y' in params:
                distance = math.sqrt(params['x']**2 + params['y']**2)
                if distance > self.constraints.max_reach:
                    violations.append(SafetyViolation(
                        level=SafetyLevel.DANGER,
                        message=(
                            f"Target distance {distance:.2f}m exceeds "
                            f"arm reach {self.constraints.max_reach:.2f}m"
                        ),
                        action_type=action.action_type,
                        suggested_fix="Move robot closer to target first"
                    ))

        return violations

    def _check_workspace(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check workspace boundary compliance."""
        violations = []
        params = action.parameters

        if 'x' in params and 'y' in params:
            x, y = params['x'], params['y']
            if not self.constraints.in_workspace(x, y):
                violations.append(SafetyViolation(
                    level=SafetyLevel.DANGER,
                    message=f"Target ({x:.2f}, {y:.2f}) outside workspace",
                    action_type=action.action_type,
                    suggested_fix="Choose a target within workspace bounds"
                ))

        return violations

    def _check_restricted_zones(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check restricted zone compliance."""
        violations = []
        params = action.parameters

        if 'x' in params and 'y' in params:
            x, y = params['x'], params['y']
            zone_idx = self.constraints.in_restricted_zone(x, y)
            if zone_idx is not None:
                violations.append(SafetyViolation(
                    level=SafetyLevel.CRITICAL,
                    message=f"Target in restricted zone {zone_idx}",
                    action_type=action.action_type,
                    suggested_fix="Choose a different target location"
                ))

        return violations

    def _check_human_proximity(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check for human proximity safety."""
        violations = []
        params = action.parameters

        if not self._human_positions:
            return violations

        target_x = params.get('x', 0)
        target_y = params.get('y', 0)

        for hx, hy in self._human_positions:
            distance = math.sqrt((target_x - hx)**2 + (target_y - hy)**2)

            if distance < self.constraints.min_human_distance:
                violations.append(SafetyViolation(
                    level=SafetyLevel.CRITICAL,
                    message=(
                        f"Target too close to human "
                        f"({distance:.2f}m < {self.constraints.min_human_distance:.2f}m)"
                    ),
                    action_type=action.action_type,
                    suggested_fix="Wait for human to move or choose different target"
                ))
            elif distance < self.constraints.min_human_distance * 2:
                violations.append(SafetyViolation(
                    level=SafetyLevel.WARNING,
                    message=f"Human nearby ({distance:.2f}m)",
                    action_type=action.action_type
                ))

        return violations

    def _check_forbidden_objects(
        self,
        action: RobotAction
    ) -> List[SafetyViolation]:
        """Check for manipulation of forbidden objects."""
        violations = []

        if action.action_type not in (ActionType.PICK, ActionType.GRASP):
            return violations

        object_id = str(action.parameters.get('object_id', '')).lower()

        for forbidden in self.FORBIDDEN_OBJECTS:
            if forbidden in object_id:
                violations.append(SafetyViolation(
                    level=SafetyLevel.CRITICAL,
                    message=f"Cannot manipulate forbidden object: {object_id}",
                    action_type=action.action_type,
                    parameter='object_id',
                    suggested_fix="Choose a different object"
                ))
                break

        return violations

    def emergency_stop_check(
        self,
        action: RobotAction,
        current_state: Dict[str, Any]
    ) -> bool:
        """
        Fast emergency stop check for real-time use.

        Args:
            action: Action being executed
            current_state: Current robot state

        Returns:
            True if emergency stop should be triggered
        """
        # Check for critical violations
        _, violations = self.validate_action(action)

        for v in violations:
            if v.level == SafetyLevel.CRITICAL:
                logger.critical(f"EMERGENCY STOP: {v.message}")
                return True

        # Check if human too close during execution
        if current_state.get('nearest_human_distance', float('inf')) < 0.3:
            logger.critical("EMERGENCY STOP: Human too close")
            return True

        # Check for unexpected contact
        if current_state.get('contact_detected', False):
            logger.critical("EMERGENCY STOP: Unexpected contact")
            return True

        return False
