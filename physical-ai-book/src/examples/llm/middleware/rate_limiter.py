#!/usr/bin/env python3
"""
Rate Limiter for LLM Requests
Physical AI Book - Chapter 6: LLM Integration

Token bucket rate limiter for controlling LLM API request rates.
Prevents hitting rate limits and manages token usage.

Features:
- Token bucket algorithm
- Burst handling
- Async support
- Token tracking

Usage:
    from physical_ai_examples.llm.middleware import RateLimiter

    # Create rate limiter (10 requests per second, burst of 20)
    limiter = RateLimiter(rate=10, burst=20)

    # Wait for token before making request
    await limiter.acquire()
    response = await llm.generate(prompt)

Author: Physical AI Book
License: MIT
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    The bucket fills at a constant rate up to a maximum capacity.
    Each request consumes tokens from the bucket.

    Attributes:
        rate: Tokens added per second
        capacity: Maximum tokens in bucket
        tokens: Current token count
        last_update: Last time tokens were added
    """
    rate: float
    capacity: float
    tokens: float = field(init=False)
    last_update: float = field(init=False)

    def __post_init__(self):
        """Initialize bucket to full capacity."""
        self.tokens = self.capacity
        self.last_update = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: float = 1.0) -> float:
        """
        Calculate time until tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if immediately available)
        """
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.rate


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Supports multiple buckets for different resource types
    (requests, tokens, etc.).
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: float = 20.0,
        *,
        token_rate: Optional[float] = None,
        token_burst: Optional[float] = None
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second
            burst: Maximum burst size
            token_rate: LLM tokens per second (optional)
            token_burst: Maximum token burst (optional)
        """
        self.request_bucket = TokenBucket(rate=rate, capacity=burst)

        if token_rate and token_burst:
            self.token_bucket = TokenBucket(rate=token_rate, capacity=token_burst)
        else:
            self.token_bucket = None

        self._lock = asyncio.Lock()
        self._stats = {
            'requests': 0,
            'waited_time': 0.0,
            'tokens_used': 0,
        }

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire permission to make a request.

        Blocks until rate limit allows the request.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            wait_time = self.request_bucket.time_until_available(tokens)
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                self._stats['waited_time'] += wait_time
                await asyncio.sleep(wait_time)

            self.request_bucket.try_acquire(tokens)
            self._stats['requests'] += 1

    async def acquire_tokens(self, token_count: int) -> None:
        """
        Acquire permission based on LLM token count.

        Args:
            token_count: Number of LLM tokens being used
        """
        if self.token_bucket is None:
            return

        async with self._lock:
            wait_time = self.token_bucket.time_until_available(token_count)
            if wait_time > 0:
                logger.debug(f"Token rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            self.token_bucket.try_acquire(token_count)
            self._stats['tokens_used'] += token_count

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired successfully
        """
        if self.request_bucket.try_acquire(tokens):
            self._stats['requests'] += 1
            return True
        return False

    @property
    def stats(self) -> Dict:
        """Get usage statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            'requests': 0,
            'waited_time': 0.0,
            'tokens_used': 0,
        }


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on API responses.

    Backs off on rate limit errors and speeds up when successful.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 100.0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            initial_rate: Starting rate
            min_rate: Minimum rate after backoff
            max_rate: Maximum rate
            backoff_factor: Multiply rate by this on error
            recovery_factor: Multiply rate by this on success
        """
        super().__init__(rate=initial_rate, burst=initial_rate * 2)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self._current_rate = initial_rate
        self._consecutive_success = 0
        self._consecutive_errors = 0

    def on_success(self) -> None:
        """Report successful request."""
        self._consecutive_success += 1
        self._consecutive_errors = 0

        # Speed up after consecutive successes
        if self._consecutive_success >= 10:
            new_rate = min(
                self._current_rate * self.recovery_factor,
                self.max_rate
            )
            if new_rate != self._current_rate:
                self._current_rate = new_rate
                self.request_bucket.rate = new_rate
                logger.debug(f"Rate increased to {new_rate:.2f}/s")
            self._consecutive_success = 0

    def on_rate_limit_error(self) -> None:
        """Report rate limit error."""
        self._consecutive_errors += 1
        self._consecutive_success = 0

        # Back off
        new_rate = max(
            self._current_rate * self.backoff_factor,
            self.min_rate
        )
        if new_rate != self._current_rate:
            self._current_rate = new_rate
            self.request_bucket.rate = new_rate
            logger.warning(f"Rate limited, backing off to {new_rate:.2f}/s")

    @property
    def current_rate(self) -> float:
        """Get current rate."""
        return self._current_rate
