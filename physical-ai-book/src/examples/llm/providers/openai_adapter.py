#!/usr/bin/env python3
"""
OpenAI LLM Provider Adapter
Physical AI Book - Chapter 6: LLM Integration

Implementation of the OpenAI provider for cloud-based LLM inference.
Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.

Features:
- Synchronous and streaming generation
- JSON mode support (with compatible models)
- Function calling
- Health checking

Usage:
    from physical_ai_examples.llm.providers import OpenAIProvider
    from physical_ai_examples.llm.core import LLMConfig

    config = LLMConfig(
        provider='openai',
        model='gpt-4',
        api_key='your-api-key'
    )
    provider = OpenAIProvider(config)
    response = await provider.generate("What is ROS 2?")

Environment:
    OPENAI_API_KEY: Your OpenAI API key

Dependencies:
    - httpx (async HTTP client)
    - openai (optional, for native SDK)

Author: Physical AI Book
License: MIT
"""

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


@LLMProvider.register('openai')
class OpenAIProvider(LLMProvider):
    """
    OpenAI provider for cloud LLM inference.

    Connects to OpenAI API (or compatible endpoints like Azure OpenAI).

    Attributes:
        api_key: OpenAI API key
        base_url: API base URL
        client: HTTP client for API requests
    """

    DEFAULT_URL = "https://api.openai.com/v1"

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration

        Raises:
            ValueError: If API key is not provided
        """
        super().__init__(config)

        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key in config."
            )

        self.base_url = config.api_base or os.getenv(
            "OPENAI_API_BASE", self.DEFAULT_URL
        )
        self._client: Optional['httpx.AsyncClient'] = None

    async def _get_client(self) -> 'httpx.AsyncClient':
        """Get or create HTTP client."""
        if httpx is None:
            raise ImportError(
                "httpx is required for OpenAIProvider. "
                "Install with: pip install httpx"
            )
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout)
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using OpenAI API.

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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if json_mode and "gpt-4" in self.config.model:
            request_data["response_format"] = {"type": "json_object"}

        try:
            response = await client.post(
                "/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"],
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = await client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"],
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data
            )

        except Exception as e:
            logger.error(f"OpenAI generation with history failed: {e}")
            raise

    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        function_call: str = "auto",
        **kwargs
    ) -> LLMResponse:
        """
        Generate with function calling support.

        Args:
            prompt: User prompt
            functions: List of function definitions
            system_prompt: Optional system prompt
            function_call: "auto", "none", or {"name": "function_name"}
            **kwargs: Additional parameters

        Returns:
            LLMResponse (may include function_call in raw_response)
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "functions": functions,
            "function_call": function_call,
        }

        try:
            response = await client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]

            content = message.get("content", "")
            if message.get("function_call"):
                content = json.dumps(message["function_call"])

            return LLMResponse(
                content=content,
                model=data["model"],
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason", "function_call"),
                raw_response=data
            )

        except Exception as e:
            logger.error(f"OpenAI function calling failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.

        Returns:
            True if API is healthy
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model IDs
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            if response.status_code == 200:
                models = response.json().get('data', [])
                return [m['id'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
