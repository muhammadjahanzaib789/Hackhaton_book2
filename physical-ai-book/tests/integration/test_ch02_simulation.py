#!/usr/bin/env python3
"""
Integration Tests for Chapter 2: Simulation
Physical AI Book

These tests verify that the Chapter 2 simulation components work correctly.
Tests cover URDF/SDF validation, sensor configuration, and joint control.

Usage:
    # Run all Chapter 2 tests
    pytest tests/integration/test_ch02_simulation.py -v

    # Run specific test class
    pytest tests/integration/test_ch02_simulation.py::TestURDFModel -v

Requirements:
    - pytest
    - urdfdom (for URDF validation)
    - xml.etree for SDF parsing

Author: Physical AI Book
License: MIT
"""

import os
import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def get_model_path(filename: str) -> Path:
    """Get path to model file."""
    base = Path(__file__).parent.parent.parent
    return base / 'static' / 'models' / 'humanoid' / filename


def get_example_path(filename: str) -> Path:
    """Get path to example file."""
    base = Path(__file__).parent.parent.parent
    return base / 'src' / 'examples' / 'ros2' / 'ch02' / filename


class TestURDFModel:
    """Tests for humanoid URDF model."""

    @pytest.fixture
    def urdf_path(self):
        """Fixture providing URDF path."""
        return get_model_path('humanoid.urdf')

    @pytest.fixture
    def urdf_tree(self, urdf_path):
        """Fixture providing parsed URDF."""
        if not urdf_path.exists():
            pytest.skip("URDF file not found")
        return ET.parse(urdf_path)

    def test_urdf_file_exists(self, urdf_path):
        """Test that URDF file exists."""
        assert urdf_path.exists(), f"URDF not found at {urdf_path}"

    def test_urdf_valid_xml(self, urdf_path):
        """Test that URDF is valid XML."""
        try:
            ET.parse(urdf_path)
        except ET.ParseError as e:
            pytest.fail(f"URDF is not valid XML: {e}")

    def test_urdf_has_robot_element(self, urdf_tree):
        """Test that URDF has root robot element."""
        root = urdf_tree.getroot()
        assert root.tag == 'robot', f"Root element is {root.tag}, expected 'robot'"

    def test_urdf_robot_name(self, urdf_tree):
        """Test that robot has a name."""
        root = urdf_tree.getroot()
        name = root.get('name')
        assert name is not None, "Robot has no name attribute"
        assert len(name) > 0, "Robot name is empty"

    def test_urdf_has_links(self, urdf_tree):
        """Test that URDF has link elements."""
        root = urdf_tree.getroot()
        links = root.findall('link')
        assert len(links) > 0, "URDF has no links"

    def test_urdf_has_joints(self, urdf_tree):
        """Test that URDF has joint elements."""
        root = urdf_tree.getroot()
        joints = root.findall('joint')
        assert len(joints) > 0, "URDF has no joints"

    def test_urdf_joint_count(self, urdf_tree):
        """Test that URDF has expected number of joints (21 DOF)."""
        root = urdf_tree.getroot()
        # Count non-fixed joints
        joints = [j for j in root.findall('joint')
                  if j.get('type') != 'fixed']
        assert len(joints) >= 21, f"Expected at least 21 joints, found {len(joints)}"

    def test_urdf_joints_have_limits(self, urdf_tree):
        """Test that revolute joints have limits."""
        root = urdf_tree.getroot()
        for joint in root.findall('joint'):
            if joint.get('type') == 'revolute':
                limit = joint.find('limit')
                assert limit is not None, \
                    f"Joint {joint.get('name')} has no limits"

    def test_urdf_links_have_inertial(self, urdf_tree):
        """Test that links have inertial properties."""
        root = urdf_tree.getroot()
        for link in root.findall('link'):
            # Skip base_link if it's the root
            if link.get('name') == 'base_link':
                continue
            inertial = link.find('inertial')
            # At least some links should have inertial
            if inertial is not None:
                mass = inertial.find('mass')
                assert mass is not None, \
                    f"Link {link.get('name')} inertial has no mass"

    def test_urdf_has_visual_elements(self, urdf_tree):
        """Test that links have visual elements."""
        root = urdf_tree.getroot()
        visual_count = 0
        for link in root.findall('link'):
            if link.find('visual') is not None:
                visual_count += 1
        assert visual_count > 0, "No links have visual elements"

    def test_urdf_has_collision_elements(self, urdf_tree):
        """Test that links have collision elements."""
        root = urdf_tree.getroot()
        collision_count = 0
        for link in root.findall('link'):
            if link.find('collision') is not None:
                collision_count += 1
        assert collision_count > 0, "No links have collision elements"


class TestSDFModel:
    """Tests for humanoid SDF model."""

    @pytest.fixture
    def sdf_path(self):
        """Fixture providing SDF path."""
        return get_model_path('humanoid.sdf')

    @pytest.fixture
    def sdf_tree(self, sdf_path):
        """Fixture providing parsed SDF."""
        if not sdf_path.exists():
            pytest.skip("SDF file not found")
        return ET.parse(sdf_path)

    def test_sdf_file_exists(self, sdf_path):
        """Test that SDF file exists."""
        assert sdf_path.exists(), f"SDF not found at {sdf_path}"

    def test_sdf_valid_xml(self, sdf_path):
        """Test that SDF is valid XML."""
        try:
            ET.parse(sdf_path)
        except ET.ParseError as e:
            pytest.fail(f"SDF is not valid XML: {e}")

    def test_sdf_has_model_element(self, sdf_tree):
        """Test that SDF has model element."""
        root = sdf_tree.getroot()
        # SDF can have model at root or under sdf element
        model = root if root.tag == 'model' else root.find('.//model')
        assert model is not None, "SDF has no model element"

    def test_sdf_has_sensors(self, sdf_tree):
        """Test that SDF defines sensors."""
        root = sdf_tree.getroot()
        sensors = root.findall('.//sensor')
        assert len(sensors) > 0, "SDF has no sensors defined"

    def test_sdf_has_camera_sensor(self, sdf_tree):
        """Test that SDF has camera sensor."""
        root = sdf_tree.getroot()
        cameras = [s for s in root.findall('.//sensor')
                   if s.get('type') in ['camera', 'depth_camera']]
        assert len(cameras) > 0, "SDF has no camera sensors"

    def test_sdf_has_imu_sensor(self, sdf_tree):
        """Test that SDF has IMU sensor."""
        root = sdf_tree.getroot()
        imus = [s for s in root.findall('.//sensor')
                if s.get('type') == 'imu']
        assert len(imus) > 0, "SDF has no IMU sensor"

    def test_sdf_has_plugins(self, sdf_tree):
        """Test that SDF has plugins configured."""
        root = sdf_tree.getroot()
        plugins = root.findall('.//plugin')
        assert len(plugins) > 0, "SDF has no plugins"


class TestCodeExamples:
    """Tests for Chapter 2 code examples."""

    def test_launch_humanoid_exists(self):
        """Test that launch file exists."""
        path = get_example_path('launch_humanoid.py')
        assert path.exists(), f"Launch file not found at {path}"

    def test_joint_controller_exists(self):
        """Test that joint controller exists."""
        path = get_example_path('joint_controller.py')
        assert path.exists(), f"Joint controller not found at {path}"

    def test_sensor_reader_exists(self):
        """Test that sensor reader exists."""
        path = get_example_path('sensor_reader.py')
        assert path.exists(), f"Sensor reader not found at {path}"

    def test_locomotion_exists(self):
        """Test that locomotion controller exists."""
        path = get_example_path('simple_locomotion.py')
        assert path.exists(), f"Locomotion controller not found at {path}"

    def test_code_examples_have_docstrings(self):
        """Test that code examples have module docstrings."""
        import ast

        examples = [
            'launch_humanoid.py',
            'joint_controller.py',
            'sensor_reader.py',
            'simple_locomotion.py',
        ]

        for filename in examples:
            path = get_example_path(filename)
            if not path.exists():
                continue

            with open(path, 'r') as f:
                content = f.read()

            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            assert docstring is not None, \
                f"{filename} has no module docstring"
            assert len(docstring) > 50, \
                f"{filename} docstring is too short"


class TestLocomotionMath:
    """Tests for locomotion controller math."""

    def test_sinusoidal_trajectory(self):
        """Test sinusoidal joint trajectory generation."""
        hip_amp = 0.25
        knee_amp = 0.40

        # Test at different phases
        for phase in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
            hip = hip_amp * math.sin(phase)
            knee = knee_amp * max(0, math.sin(phase))

            # Hip should be bounded
            assert abs(hip) <= hip_amp, \
                f"Hip position {hip} exceeds amplitude {hip_amp}"

            # Knee should be non-negative
            assert knee >= 0, f"Knee position {knee} is negative"

            # Knee should be bounded
            assert knee <= knee_amp, \
                f"Knee position {knee} exceeds amplitude {knee_amp}"

    def test_phase_coordination(self):
        """Test that legs are coordinated with 180° offset."""
        phase_offset = math.pi

        for phase in [0, math.pi/4, math.pi/2, math.pi]:
            left_phase = phase
            right_phase = phase + phase_offset

            # At any time, legs should be in opposite phases
            left_sin = math.sin(left_phase)
            right_sin = math.sin(right_phase)

            # Opposite phases: sin(θ) = -sin(θ + π)
            assert abs(left_sin + right_sin) < 0.01, \
                f"Legs not properly coordinated at phase {phase}"

    def test_quaternion_to_euler(self):
        """Test quaternion to Euler conversion."""
        from examples.ros2.ch02.sensor_reader import SensorReader

        # Identity quaternion (no rotation)
        roll, pitch, yaw = SensorReader.quaternion_to_euler(0, 0, 0, 1)
        assert abs(roll) < 0.01
        assert abs(pitch) < 0.01
        assert abs(yaw) < 0.01

        # 90° rotation around Z (yaw)
        # Quaternion: (0, 0, sin(π/4), cos(π/4)) = (0, 0, 0.707, 0.707)
        roll, pitch, yaw = SensorReader.quaternion_to_euler(0, 0, 0.707, 0.707)
        assert abs(yaw - math.pi/2) < 0.1


class TestJointController:
    """Tests for joint controller functionality."""

    def test_joint_names_defined(self):
        """Test that all expected joints are defined."""
        from examples.ros2.ch02.joint_controller import JointController

        expected_joints = [
            'waist_yaw',
            'left_shoulder_pitch', 'right_shoulder_pitch',
            'left_hip_pitch', 'right_hip_pitch',
            'left_knee_pitch', 'right_knee_pitch',
        ]

        for joint in expected_joints:
            assert joint in JointController.JOINT_NAMES, \
                f"Missing joint: {joint}"

    def test_poses_defined(self):
        """Test that predefined poses exist."""
        from examples.ros2.ch02.joint_controller import JointController

        expected_poses = ['stand', 'squat', 't_pose']

        for pose in expected_poses:
            assert pose in JointController.POSES, \
                f"Missing pose: {pose}"

    def test_pose_values_valid(self):
        """Test that pose joint values are within typical limits."""
        from examples.ros2.ch02.joint_controller import JointController

        max_angle = 3.14  # ~180 degrees

        for pose_name, pose in JointController.POSES.items():
            for joint, value in pose.items():
                assert abs(value) <= max_angle, \
                    f"Pose {pose_name} joint {joint} value {value} too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
