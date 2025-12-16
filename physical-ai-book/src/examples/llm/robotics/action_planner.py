#!/usr/bin/env python3
"""
LLM Action Planner for Robotics
Physical AI Book - Chapter 6: LLM Integration

Translates natural language commands into executable robot action plans.
Integrates LLM reasoning with robot capabilities and constraints.

Features:
- Command understanding
- Task decomposition
- Action sequencing
- Plan validation
- Recovery planning

Usage:
    from physical_ai_examples.llm.robotics import ActionPlanner
    from physical_ai_examples.llm.core import LLMConfig, LLMProvider

    config = LLMConfig(provider='ollama', model='llama3.2')
    provider = LLMProvider.create(config)
    planner = ActionPlanner(provider)

    plan = await planner.plan("Pick up the red cup and bring it to me")
    for action in plan.actions:
        print(f"Action: {action.action_type.value}")

Author: Physical AI Book
License: MIT
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.provider import LLMConfig, LLMProvider, LLMResponse
from ..core.schemas import (
    ActionType,
    ActionResponse,
    ActionSchema,
    RobotAction,
    STANDARD_SCHEMAS
)

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    """
    Current state of the robot's world.

    Used to provide context for action planning.

    Attributes:
        robot_position: Current robot (x, y, theta)
        held_object: Object currently held (None if empty)
        visible_objects: Objects detected by vision
        known_locations: Named locations and their coordinates
        battery_level: Battery percentage
        arm_state: Arm status (ready, busy, error)
    """
    robot_position: tuple = (0.0, 0.0, 0.0)
    held_object: Optional[str] = None
    visible_objects: List[Dict[str, Any]] = field(default_factory=list)
    known_locations: Dict[str, tuple] = field(default_factory=dict)
    battery_level: float = 100.0
    arm_state: str = "ready"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompting."""
        return {
            'robot_position': {
                'x': self.robot_position[0],
                'y': self.robot_position[1],
                'theta': self.robot_position[2]
            },
            'held_object': self.held_object,
            'visible_objects': self.visible_objects,
            'known_locations': {
                name: {'x': pos[0], 'y': pos[1]}
                for name, pos in self.known_locations.items()
            },
            'battery_level': self.battery_level,
            'arm_state': self.arm_state,
        }


class ActionPlanner:
    """
    LLM-based action planner for robot task execution.

    Converts natural language commands into sequences of
    executable robot actions.
    """

    # System prompt for action planning
    SYSTEM_PROMPT = """You are a robot action planner. Convert natural language commands into sequences of robot actions.

AVAILABLE ACTIONS:
{actions}

OUTPUT FORMAT:
Always respond with a JSON object:
{{
    "goal": "Summary of what the user wants",
    "plan": [
        {{"action_type": "action_name", "parameters": {{...}}, "description": "Why this action"}}
    ],
    "reasoning": "Brief explanation of the plan",
    "confidence": 0.0-1.0
}}

RULES:
1. Only use actions from the AVAILABLE ACTIONS list
2. Include all required parameters for each action
3. Order actions logically (navigate before pick, etc.)
4. Consider the current world state
5. Set confidence based on plan certainty

If the request cannot be fulfilled, respond with:
{{
    "goal": "User's request",
    "plan": [],
    "reasoning": "Why it cannot be done",
    "confidence": 0.0
}}"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        action_schemas: Optional[Dict[ActionType, ActionSchema]] = None,
        max_actions: int = 10,
        min_confidence: float = 0.3
    ):
        """
        Initialize action planner.

        Args:
            llm_provider: LLM provider for planning
            action_schemas: Custom action schemas
            max_actions: Maximum actions in a plan
            min_confidence: Minimum confidence to accept plan
        """
        self.llm = llm_provider
        self.schemas = action_schemas or STANDARD_SCHEMAS
        self.max_actions = max_actions
        self.min_confidence = min_confidence

    def _format_actions_for_prompt(self) -> str:
        """Format available actions for system prompt."""
        lines = []
        for action_type, schema in self.schemas.items():
            params = []
            for name, param in schema.parameters.items():
                req = "(required)" if param.required else "(optional)"
                params.append(f"{name}: {param.param_type} {req}")

            lines.append(
                f"- {action_type.value}: {schema.description}\n"
                f"  Parameters: {', '.join(params) or 'none'}"
            )
        return "\n".join(lines)

    async def plan(
        self,
        command: str,
        world_state: Optional[WorldState] = None,
        **kwargs
    ) -> ActionResponse:
        """
        Generate an action plan for a natural language command.

        Args:
            command: Natural language command
            world_state: Current world state
            **kwargs: Additional parameters

        Returns:
            ActionResponse with planned actions
        """
        world_state = world_state or WorldState()

        # Build prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            actions=self._format_actions_for_prompt()
        )

        user_prompt = f"""WORLD STATE:
{json.dumps(world_state.to_dict(), indent=2)}

USER COMMAND: {command}

Generate an action plan:"""

        # Get LLM response
        try:
            response = await self.llm.generate(
                user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            logger.debug(f"LLM response: {response.content}")

            # Parse response
            plan = self._parse_response(response.content)

            # Validate plan
            if plan.confidence < self.min_confidence:
                logger.warning(
                    f"Plan confidence {plan.confidence} below threshold "
                    f"{self.min_confidence}"
                )

            if len(plan.plan) > self.max_actions:
                logger.warning(
                    f"Plan has {len(plan.plan)} actions, "
                    f"truncating to {self.max_actions}"
                )
                plan.plan = plan.plan[:self.max_actions]

            return plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return ActionResponse(
                goal=command,
                plan=[],
                reasoning=f"Planning failed: {str(e)}",
                confidence=0.0
            )

    def _parse_response(self, response: str) -> ActionResponse:
        """
        Parse LLM response into ActionResponse.

        Args:
            response: Raw LLM response

        Returns:
            ActionResponse object
        """
        try:
            # Extract JSON
            data = json.loads(response)

            # Parse actions
            actions = []
            for action_data in data.get('plan', []):
                try:
                    action_type = ActionType(action_data['action_type'])
                    action = RobotAction(
                        action_type=action_type,
                        parameters=action_data.get('parameters', {}),
                        metadata={
                            'description': action_data.get('description', '')
                        }
                    )
                    actions.append(action)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid action: {e}")

            return ActionResponse(
                goal=data.get('goal', ''),
                plan=actions,
                reasoning=data.get('reasoning', ''),
                confidence=float(data.get('confidence', 0.5))
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return ActionResponse(
                goal='',
                plan=[],
                reasoning=f"Parse error: {str(e)}",
                confidence=0.0
            )

    async def replan_on_failure(
        self,
        original_plan: ActionResponse,
        failed_action: RobotAction,
        failure_reason: str,
        world_state: WorldState
    ) -> ActionResponse:
        """
        Generate a recovery plan after action failure.

        Args:
            original_plan: The original plan
            failed_action: Action that failed
            failure_reason: Why it failed
            world_state: Current world state

        Returns:
            New ActionResponse with recovery plan
        """
        recovery_prompt = f"""ORIGINAL GOAL: {original_plan.goal}

FAILED ACTION: {failed_action.action_type.value}
FAILURE REASON: {failure_reason}

CURRENT WORLD STATE:
{json.dumps(world_state.to_dict(), indent=2)}

Generate a RECOVERY plan to achieve the original goal.
Consider alternative approaches that avoid the failed action.
If recovery is not possible, explain why.

Recovery plan:"""

        try:
            response = await self.llm.generate(
                recovery_prompt,
                system_prompt=self.SYSTEM_PROMPT.format(
                    actions=self._format_actions_for_prompt()
                ),
                json_mode=True
            )

            plan = self._parse_response(response.content)
            plan.goal = f"Recovery: {original_plan.goal}"
            return plan

        except Exception as e:
            logger.error(f"Recovery planning failed: {e}")
            return ActionResponse(
                goal=f"Recovery: {original_plan.goal}",
                plan=[],
                reasoning=f"Recovery planning failed: {str(e)}",
                confidence=0.0
            )

    async def decompose_task(
        self,
        high_level_task: str,
        world_state: Optional[WorldState] = None
    ) -> List[str]:
        """
        Decompose a high-level task into subtasks.

        Useful for complex tasks that need multi-step execution.

        Args:
            high_level_task: Complex task description
            world_state: Current world state

        Returns:
            List of subtask descriptions
        """
        decompose_prompt = f"""TASK: {high_level_task}

Break this task into simple, sequential subtasks.
Each subtask should be achievable with a single action or short action sequence.

Respond with JSON:
{{
    "subtasks": [
        "First subtask description",
        "Second subtask description",
        ...
    ]
}}

Subtasks:"""

        try:
            response = await self.llm.generate(
                decompose_prompt,
                json_mode=True
            )

            data = json.loads(response.content)
            return data.get('subtasks', [high_level_task])

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return [high_level_task]
