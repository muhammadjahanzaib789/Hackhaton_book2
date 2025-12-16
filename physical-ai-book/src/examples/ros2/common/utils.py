#!/usr/bin/env python3
"""
Utility Functions for Physical AI Book Examples
Common helper functions used across ROS 2 examples.

Author: Physical AI Book
License: MIT
"""

import functools
import time
from typing import Any, Callable, Optional, TypeVar
import threading

T = TypeVar('T')


def rate_limit(min_interval: float):
    """
    Decorator to rate-limit function calls.

    Args:
        min_interval: Minimum seconds between calls

    Usage:
        @rate_limit(0.1)  # Max 10 calls per second
        def my_callback(msg):
            process(msg)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        last_call = [0.0]
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            with lock:
                now = time.time()
                if now - last_call[0] >= min_interval:
                    last_call[0] = now
                    return func(*args, **kwargs)
                return None
        return wrapper
    return decorator


def with_timeout(timeout_sec: float, default: Any = None):
    """
    Decorator to add timeout to blocking functions.

    Args:
        timeout_sec: Maximum execution time in seconds
        default: Value to return on timeout

    Usage:
        @with_timeout(5.0)
        def long_operation():
            # May take a while
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = [default]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout_sec)

            if thread.is_alive():
                return default

            if exception[0] is not None:
                raise exception[0]

            return result[0]
        return wrapper
    return decorator


def safe_callback(logger=None):
    """
    Decorator to wrap callbacks with exception handling.

    Args:
        logger: Optional ROS 2 logger for error messages

    Usage:
        @safe_callback(self.get_logger())
        def callback(msg):
            # May raise exceptions
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f'Exception in {func.__name__}: {e}')
                return None
        return wrapper
    return decorator


def get_parameter_or_default(node, name: str, default: Any) -> Any:
    """
    Safely get a ROS 2 parameter with a default value.

    Args:
        node: ROS 2 node instance
        name: Parameter name
        default: Default value if parameter not set

    Returns:
        Parameter value or default
    """
    try:
        if node.has_parameter(name):
            return node.get_parameter(name).value
        else:
            node.declare_parameter(name, default)
            return default
    except Exception:
        return default


class RateLimiter:
    """
    Class-based rate limiter for more control.

    Usage:
        limiter = RateLimiter(rate_hz=10)

        def callback(msg):
            if limiter.should_process():
                # Process message
                pass
    """

    def __init__(self, rate_hz: float):
        """
        Initialize rate limiter.

        Args:
            rate_hz: Maximum rate in Hz
        """
        self.min_interval = 1.0 / rate_hz
        self.last_time = 0.0
        self._lock = threading.Lock()

    def should_process(self) -> bool:
        """
        Check if enough time has passed for next call.

        Returns:
            True if caller should proceed, False to skip
        """
        with self._lock:
            now = time.time()
            if now - self.last_time >= self.min_interval:
                self.last_time = now
                return True
            return False

    def reset(self):
        """Reset the rate limiter."""
        with self._lock:
            self.last_time = 0.0


class MovingAverage:
    """
    Simple moving average calculator for sensor data.

    Usage:
        avg = MovingAverage(window_size=10)
        smoothed = avg.update(sensor_reading)
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize moving average.

        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.values: list = []

    def update(self, value: float) -> float:
        """
        Add a new value and return the average.

        Args:
            value: New sample value

        Returns:
            Current moving average
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

    def reset(self):
        """Clear all stored values."""
        self.values.clear()

    @property
    def current_average(self) -> float:
        """Get current average without adding new value."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-pi, pi]
    """
    import math
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
