#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer
Physical AI Book - Chapter 6: LLM Integration

Provides a unified interface for different LLM providers,
enabling seamless switching between Ollama, OpenAI, and others.

Design Pattern: Strategy Pattern with Factory
- LLMProvider: Abstract base class defining the interface
- Concrete providers: Ollama, OpenAI, etc.
- LLMProviderFactory: Creates provider instances

Usage:
    from physical_ai_examples.llm.core import LLMProvider, LLMConfig

    config = LLMConfig(provider='ollama', model='llama3.2')
    provider = LLMProvider.create(config)
    response = await provider.generate("Plan a task to pick up the red cup")

Author: Physical AI Book
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
import json
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """
    Configuration for LLM providers.

    Attributes:
        provider: Provider type (ollama, openai, etc.)
        model: Model name/identifier
        api_key: API key for cloud providers (optional)
        api_base: Base URL for API (optional, for self-hosted)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        extra: Additional provider-specific settings
    """
    provider: str = "ollama"
    model: str = "llama3.2"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.

    Attributes:
        content: Generated text content
        model: Model that generated the response
        usage: Token usage statistics
        finish_reason: Why generation stopped
        raw_response: Original provider response
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get('total_tokens', 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'model': self.model,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the required abstract methods.

    Example:
        class MyProvider(LLMProvider):
            async def generate(self, prompt, **kwargs):
                # Implementation
                pass
    """

    # Registry of provider implementations
    _providers: Dict[str, Type['LLMProvider']] = {}

    def __init__(self, config: LLMConfig):
        """
        Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._initialized = False

    @classmethod
    def register(cls, provider_type: str):
        """
        Decorator to register a provider implementation.

        Args:
            provider_type: Provider type identifier

        Usage:
            @LLMProvider.register('ollama')
            class OllamaProvider(LLMProvider):
                pass
        """
        def decorator(provider_cls: Type['LLMProvider']):
            cls._providers[provider_type.lower()] = provider_cls
            return provider_cls
        return decorator

    @classmethod
    def create(cls, config: LLMConfig) -> 'LLMProvider':
        """
        Factory method to create a provider instance.

        Args:
            config: Provider configuration

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_type = config.provider.lower()
        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider_type}'. "
                f"Available: {available}"
            )
        return cls._providers[provider_type](config)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: Whether to request JSON output
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Yields:
            Generated text chunks
        """
        pass

    async def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON response.

        Args:
            prompt: User prompt requesting JSON output
            schema: Optional JSON schema for validation
            **kwargs: Additional arguments

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If response doesn't match schema
        """
        response = await self.generate(prompt, json_mode=True, **kwargs)

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {response.content}")
            raise

        if schema:
            self._validate_schema(result, schema)

        return result

    def _validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> None:
        """
        Validate data against a JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema

        Raises:
            ValueError: If validation fails
        """
        # Basic validation - can be extended with jsonschema library
        required = schema.get('required', [])
        properties = schema.get('properties', {})

        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                if expected_type and not self._check_type(value, expected_type):
                    raise ValueError(
                        f"Field '{field}' has wrong type. "
                        f"Expected {expected_type}, got {type(value).__name__}"
                    )

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        return isinstance(value, type_map.get(expected, object))

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if provider is ready, False otherwise
        """
        pass

    async def initialize(self) -> None:
        """
        Initialize the provider (load models, connect, etc.).
        Called automatically before first use.
        """
        if not self._initialized:
            await self._do_initialize()
            self._initialized = True

    async def _do_initialize(self) -> None:
        """Override in subclasses for custom initialization."""
        pass

    async def shutdown(self) -> None:
        """Clean up resources when done."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"


# Default implementation placeholder for Ollama
# Full implementation in providers/ollama_adapter.py
@LLMProvider.register('ollama')
class OllamaProviderStub(LLMProvider):
    """
    Stub implementation - see providers/ollama_adapter.py for full version.
    This stub allows the module to load without Ollama installed.
    """

    async def generate(self, prompt, **kwargs) -> LLMResponse:
        raise NotImplementedError(
            "Full Ollama provider in llm/providers/ollama_adapter.py"
        )

    async def generate_stream(self, prompt, **kwargs):
        raise NotImplementedError(
            "Full Ollama provider in llm/providers/ollama_adapter.py"
        )

    async def health_check(self) -> bool:
        return False
