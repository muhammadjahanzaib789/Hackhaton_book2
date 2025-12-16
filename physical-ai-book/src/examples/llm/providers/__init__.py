"""
LLM Provider Implementations
Physical AI Book - Chapter 6

Provider adapters for different LLM backends.
"""

from .ollama_adapter import OllamaProvider
from .openai_adapter import OpenAIProvider

__all__ = ['OllamaProvider', 'OpenAIProvider']
