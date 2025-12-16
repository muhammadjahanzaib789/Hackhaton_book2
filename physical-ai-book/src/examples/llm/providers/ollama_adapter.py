#!/usr/bin/env python3
"""
Ollama LLM Provider Adapter
Physical AI Book - Chapter 6: LLM Integration

Full implementation of the Ollama provider for local LLM inference.
Ollama enables running models like Llama, Mistral, and others locally.

Features:
- Synchronous and streaming generation
- JSON mode support
- Health checking
- Conversation history

Usage:
    from physical_ai_examples.llm.providers import OllamaProvider
    from physical_ai_examples.llm.core import LLMConfig

    config = LLMConfig(provider='ollama', model='llama3.2')
    provider = OllamaProvider(config)
    response = await provider.generate("What is ROS 2?")

Dependencies:
    - httpx (async HTTP client)
    - ollama (optional, for native API)

Author: Physical AI Book
License: MIT
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from ..core.provider import LLMConfig, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@LLMProvider.register('ollama')
class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.

    Connects to a local Ollama server (default: http://localhost:11434).
    Supports all models available in Ollama (llama3, mistral, etc.).

    Attributes:
        base_url: Ollama server URL
        client: HTTP client for API requests
    """

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.base_url = config.api_base or os.getenv("OLLAMA_HOST", self.DEFAULT_URL)
        self._client: Optional['httpx.AsyncClient'] = None

    async def _get_client(self) -> 'httpx.AsyncClient':
        """Get or create HTTP client."""
        if httpx is None:
            raise ImportError(
                "httpx is required for OllamaProvider. "
                "Install with: pip install httpx"
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.config.timeout)
            )
        return self._client

    async def _do_initialize(self) -> None:
        """Initialize the provider."""
        client = await self._get_client()
        # Check if model is available
        try:
            response = await client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                if self.config.model not in available:
                    # Try pulling the model
                    logger.info(f"Model {self.config.model} not found, pulling...")
        except Exception as e:
            logger.warning(f"Could not check Ollama models: {e}")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: Request JSON output format
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if json_mode:
            request_data["format"] = "json"

        try:
            response = await client.post(
                "/api/chat",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["message"]["content"],
                model=self.config.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) +
                        data.get("eval_count", 0)
                    )
                },
                finish_reason="stop",
                raw_response=data
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data:
                            content = data["message"].get("content", "")
                            if content:
                                yield content
                        if data.get("done", False):
                            break

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    async def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        client = await self._get_client()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        request_data = {
            "model": self.config.model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        try:
            response = await client.post("/api/chat", json=request_data)
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
                finish_reason="stop",
                raw_response=data
            )

        except Exception as e:
            logger.error(f"Ollama generation with history failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is healthy
        """
        try:
            client = await self._get_client()
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
