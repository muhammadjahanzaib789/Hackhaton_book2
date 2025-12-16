"""
Physical AI Book - Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.

Usage:
    Fixtures defined here are automatically available to all test modules.

Author: Physical AI Book
License: MIT
"""

import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "ros2: mark test as requiring ROS 2 environment"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


@pytest.fixture(scope="session")
def ros2_available():
    """Check if ROS 2 is available in the environment."""
    try:
        import rclpy
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gazebo_available():
    """Check if Gazebo is available in the environment."""
    import shutil
    return shutil.which('gz') is not None or shutil.which('gazebo') is not None


@pytest.fixture
def temp_urdf(tmp_path):
    """Provide a temporary URDF file for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
"""
    urdf_file = tmp_path / "test_robot.urdf"
    urdf_file.write_text(urdf_content)
    return str(urdf_file)


@pytest.fixture
def example_status_json():
    """Provide example robot status JSON."""
    return {
        'x': 1.0,
        'y': 0.5,
        'theta': 0.785,
        'linear_vel': 0.3,
        'angular_vel': 0.1
    }


@pytest.fixture
def example_velocity_command():
    """Provide example velocity command values."""
    return {
        'linear_x': 0.5,
        'linear_y': 0.0,
        'linear_z': 0.0,
        'angular_x': 0.0,
        'angular_y': 0.0,
        'angular_z': 0.1
    }
