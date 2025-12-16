---
sidebar_position: 1
title: "Lesson 1: LLM Fundamentals for Robotics"
description: "Understanding language models and their application to physical AI"
---

# LLM Fundamentals for Robotics

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand how LLMs work at a high level
2. Connect to LLM APIs using a provider abstraction
3. Design prompts for robotic applications
4. Handle LLM outputs safely in physical systems

## Prerequisites

- Basic Python programming
- Understanding of APIs and HTTP requests
- Familiarity with JSON data format

## Why LLMs for Robotics?

Large Language Models enable robots to:

```
┌─────────────────────────────────────────────────────────────┐
│              LLM Capabilities for Robotics                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Natural Language Understanding                             │
│  ──────────────────────────────                             │
│  "Pick up the red cup and place it on the shelf"           │
│       │                                                     │
│       ▼                                                     │
│  [intent: pick_place, object: red_cup, target: shelf]      │
│                                                             │
│  Task Decomposition                                         │
│  ─────────────────────                                      │
│  "Clean the kitchen" → [clear_table, wash_dishes,          │
│                         wipe_counters, sweep_floor]        │
│                                                             │
│  Reasoning & Planning                                       │
│  ───────────────────────                                    │
│  "The cup is behind the box" → [move_box, grasp_cup]       │
│                                                             │
│  Error Recovery                                             │
│  ──────────────────                                         │
│  "Grasp failed" → "Try approaching from different angle"   │
│                                                             │
│  Human-Robot Dialogue                                       │
│  ────────────────────────                                   │
│  Robot: "I found two red objects. Which one?"              │
│  Human: "The one on the left"                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## LLM Architecture Overview

### How Transformers Work

```
┌─────────────────────────────────────────────────────────────┐
│                 Transformer Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "Move the robot arm to position"                    │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Tokenization                            │   │
│  │  ["Move", "the", "robot", "arm", "to", "position"]  │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Embedding Layer                         │   │
│  │  Token → Vector (d=4096)                            │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Self-Attention Layers (x N)                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Query Q │  │  Key K  │  │ Value V │            │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘            │   │
│  │       │            │            │                  │   │
│  │       └────────────┼────────────┘                  │   │
│  │                    │                               │   │
│  │           Attention(Q,K,V)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Output (Next Token Prediction)            │   │
│  │  Probability distribution over vocabulary           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description | Robotics Relevance |
|---------|-------------|-------------------|
| **Tokenization** | Breaking text into tokens | Affects prompt length limits |
| **Context Window** | Max tokens model can process | Limits history/description length |
| **Temperature** | Randomness in generation | Lower = more deterministic |
| **Top-p (Nucleus)** | Probability mass for sampling | Controls diversity |
| **System Prompt** | Instructions for behavior | Defines robot's capabilities |

## LLM Provider Abstraction

### Why Use Abstractions?

```
┌─────────────────────────────────────────────────────────────┐
│              Provider Abstraction Benefits                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Without Abstraction:                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ OpenAI Code │  │ Anthropic   │  │ Local LLM   │        │
│  │             │  │    Code     │  │    Code     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│        │                │                │                 │
│        └────────────────┼────────────────┘                 │
│                         │                                  │
│                    Duplicate logic, hard to switch         │
│                                                             │
│  With Abstraction:                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 LLMProvider Interface                │   │
│  │              generate(prompt) → response             │   │
│  └─────────────────────────────────────────────────────┘   │
│        │                │                │                 │
│        ▼                ▼                ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │  OpenAI  │    │ Anthropic│    │  Ollama  │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│                                                             │
│  → Easy to switch providers                                │
│  → Consistent interface                                    │
│  → Centralized error handling                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Provider Implementation

```python
"""
LLM Provider Abstraction
Physical AI Book - Chapter 6

Unified interface for multiple LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import os


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    system_prompt: Optional[str] = None


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface for new providers.
    """

    _registry: Dict[str, type] = {}

    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod
    def register(cls, provider_type: str):
        """Decorator to register provider implementations."""
        def decorator(provider_cls):
            cls._registry[provider_type.lower()] = provider_cls
            return provider_cls
        return decorator

    @classmethod
    def create(cls, provider_type: str, config: LLMConfig) -> 'LLMProvider':
        """Factory method to create provider instances."""
        if provider_type.lower() not in cls._registry:
            raise ValueError(f"Unknown provider: {provider_type}")
        return cls._registry[provider_type.lower()](config)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_with_history(self, messages: List[Dict[str, str]],
                              **kwargs) -> LLMResponse:
        """Generate with conversation history."""
        pass


@LLMProvider.register("ollama")
class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.

    Supports models like Llama, Mistral, etc.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using Ollama API."""
        import requests

        messages = []
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=self.config.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) +
                               data.get("eval_count", 0)
            },
            finish_reason="stop"
        )

    def generate_with_history(self, messages: List[Dict[str, str]],
                              **kwargs) -> LLMResponse:
        """Generate with conversation history."""
        import requests

        full_messages = []
        if self.config.system_prompt:
            full_messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        full_messages.extend(messages)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=self.config.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": 0
            },
            finish_reason="stop"
        )


@LLMProvider.register("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using OpenAI API."""
        import requests

        messages = []
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data["usage"],
            finish_reason=data["choices"][0]["finish_reason"]
        )

    def generate_with_history(self, messages: List[Dict[str, str]],
                              **kwargs) -> LLMResponse:
        """Generate with conversation history."""
        import requests

        full_messages = []
        if self.config.system_prompt:
            full_messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        full_messages.extend(messages)

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.config.model,
                "messages": full_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data["usage"],
            finish_reason=data["choices"][0]["finish_reason"]
        )
```

## Prompt Engineering for Robotics

### System Prompt Design

```python
ROBOT_SYSTEM_PROMPT = """You are the AI brain of a humanoid robot assistant.

CAPABILITIES:
- Navigation: move to locations, follow paths
- Manipulation: pick up objects, place objects, open doors
- Perception: identify objects, read text, recognize people
- Communication: speak, listen, display information

CONSTRAINTS:
- You can only manipulate objects within arm's reach (0.8m)
- Maximum payload: 2kg
- You cannot climb stairs (use elevator)
- Battery life: approximately 4 hours active operation

SAFETY RULES (MUST FOLLOW):
1. Never approach humans faster than 0.5 m/s
2. Stop immediately if contact detected
3. Do not manipulate sharp or hot objects
4. Always announce actions before executing

OUTPUT FORMAT:
When given a task, respond with a JSON action plan:
{
  "understanding": "Brief summary of the task",
  "steps": [
    {"action": "action_name", "params": {...}},
    ...
  ],
  "safety_checks": ["check1", "check2"],
  "estimated_time": "X seconds"
}

If you cannot complete the task, explain why and suggest alternatives."""
```

### Structured Output Prompting

```python
def create_task_prompt(user_request: str, scene_context: dict) -> str:
    """
    Create a structured prompt for task planning.

    Args:
        user_request: Natural language request
        scene_context: Current scene information

    Returns:
        Formatted prompt string
    """
    return f"""TASK REQUEST: {user_request}

CURRENT SCENE:
- Robot location: {scene_context.get('robot_position', 'unknown')}
- Visible objects: {json.dumps(scene_context.get('objects', []))}
- Nearby humans: {scene_context.get('human_count', 0)}
- Time of day: {scene_context.get('time', 'unknown')}

Please analyze this request and provide an action plan.
Remember to check safety constraints before each action.

Response (JSON only):"""


# Example usage
scene = {
    "robot_position": "kitchen_entrance",
    "objects": [
        {"name": "red_cup", "location": "table", "distance": 1.2},
        {"name": "plate", "location": "counter", "distance": 2.5}
    ],
    "human_count": 1,
    "time": "14:30"
}

prompt = create_task_prompt("Bring me the red cup", scene)
```

### Response Parsing

```python
import json
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class RobotAction:
    """Parsed robot action."""
    action: str
    params: Dict[str, Any]
    safety_checks: List[str]


def parse_llm_response(response: str) -> Optional[List[RobotAction]]:
    """
    Parse LLM response into robot actions.

    Args:
        response: Raw LLM response string

    Returns:
        List of RobotAction objects, or None if parsing fails
    """
    try:
        # Extract JSON from response
        # Handle cases where LLM adds extra text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            return None

        json_str = response[json_start:json_end]
        data = json.loads(json_str)

        actions = []
        for step in data.get('steps', []):
            actions.append(RobotAction(
                action=step['action'],
                params=step.get('params', {}),
                safety_checks=data.get('safety_checks', [])
            ))

        return actions

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse LLM response: {e}")
        return None


def validate_action(action: RobotAction,
                    available_actions: List[str]) -> bool:
    """
    Validate that an action is safe and executable.

    Args:
        action: Parsed action
        available_actions: List of valid action names

    Returns:
        True if action is valid
    """
    # Check action exists
    if action.action not in available_actions:
        return False

    # Check required parameters based on action type
    required_params = {
        'navigate': ['target'],
        'pick': ['object_id'],
        'place': ['location'],
        'speak': ['message'],
    }

    required = required_params.get(action.action, [])
    for param in required:
        if param not in action.params:
            return False

    return True
```

## Safety Considerations

### LLM Output Validation

```python
class LLMSafetyValidator:
    """
    Validate LLM outputs for robotic safety.

    Prevents execution of unsafe or invalid commands.
    """

    # Actions that require extra validation
    RESTRICTED_ACTIONS = ['grab_human', 'high_speed', 'override_safety']

    # Maximum values for parameters
    PARAM_LIMITS = {
        'speed': 0.5,        # m/s
        'force': 20.0,       # N
        'height': 1.5,       # m
        'weight': 2.0,       # kg
    }

    def validate(self, actions: List[RobotAction]) -> tuple:
        """
        Validate a sequence of actions.

        Args:
            actions: List of planned actions

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        for i, action in enumerate(actions):
            # Check for restricted actions
            if action.action in self.RESTRICTED_ACTIONS:
                errors.append(
                    f"Action {i}: '{action.action}' is restricted"
                )

            # Check parameter limits
            for param, value in action.params.items():
                if param in self.PARAM_LIMITS:
                    if isinstance(value, (int, float)):
                        if value > self.PARAM_LIMITS[param]:
                            errors.append(
                                f"Action {i}: {param}={value} exceeds "
                                f"limit {self.PARAM_LIMITS[param]}"
                            )

            # Check for dangerous object interactions
            if action.action == 'pick':
                obj = action.params.get('object_id', '')
                if any(danger in obj.lower()
                       for danger in ['knife', 'hot', 'sharp']):
                    errors.append(
                        f"Action {i}: Cannot manipulate dangerous "
                        f"object '{obj}'"
                    )

        return len(errors) == 0, errors


# Usage
validator = LLMSafetyValidator()
is_valid, errors = validator.validate(actions)

if not is_valid:
    print("Safety validation failed:")
    for error in errors:
        print(f"  - {error}")
```

### Human-in-the-Loop Confirmation

```python
class HumanConfirmation:
    """
    Require human confirmation for certain actions.

    Adds a safety layer for high-risk operations.
    """

    # Actions requiring confirmation
    CONFIRMATION_REQUIRED = [
        'navigate_to_human',
        'pick_near_human',
        'open_door',
        'leave_area',
    ]

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def needs_confirmation(self, action: RobotAction) -> bool:
        """Check if action needs human confirmation."""
        return action.action in self.CONFIRMATION_REQUIRED

    def request_confirmation(self, action: RobotAction) -> bool:
        """
        Request human confirmation for action.

        In production, would use speech/UI.
        """
        print(f"\n⚠️  CONFIRMATION REQUIRED")
        print(f"Action: {action.action}")
        print(f"Parameters: {action.params}")
        print(f"\nProceed? (yes/no): ", end="")

        # In real system, would use speech recognition or UI
        response = input().strip().lower()
        return response in ['yes', 'y', 'proceed', 'confirm']
```

## Summary

Key takeaways from this lesson:

1. **LLMs enable** natural language robot control
2. **Provider abstraction** allows switching between APIs
3. **Structured prompts** improve response quality
4. **Safety validation** is critical for physical systems
5. **Human confirmation** adds safety for risky actions

## Next Steps

In the [next lesson](./lesson-02-task-planning.md), we will:
- Implement LLM-based task decomposition
- Create action schemas for robot primitives
- Build an autonomous task executor

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Local LLMs](https://ollama.ai/)
- [LangChain for Robotics](https://python.langchain.com/)
