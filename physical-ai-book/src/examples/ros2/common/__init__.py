"""
Physical AI Book - Common ROS 2 Utilities
Shared utilities and base classes for ROS 2 examples.

This package provides:
- Base node classes with common patterns
- Message utilities
- Timing helpers
- Logging utilities

Author: Physical AI Book
License: MIT
"""

from .base_node import BasePhysicalAINode
from .utils import (
    rate_limit,
    with_timeout,
    safe_callback,
    get_parameter_or_default,
)

__version__ = '0.1.0'
__all__ = [
    'BasePhysicalAINode',
    'rate_limit',
    'with_timeout',
    'safe_callback',
    'get_parameter_or_default',
]
