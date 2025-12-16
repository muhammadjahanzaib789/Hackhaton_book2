#!/usr/bin/env python3
"""
Safety Monitor for Home Assistant Robot
Physical AI Book - Chapter 8: Capstone

Comprehensive safety monitoring system that oversees all robot
operations and enforces safety constraints.

Features:
- Real-time safety monitoring
- Multi-layer safety checks
- Emergency stop handling
- Collision avoidance
- Workspace boundary enforcement
- Battery and thermal monitoring
- Human detection and safety zones

Usage:
    ros2 run home_assistant_robot safety_monitor

Dependencies:
    - rclpy
    - sensor_msgs
    - geometry_msgs

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, Float64, String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, BatteryState
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import threading
import time
import json
import math


class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = auto()      # All systems nominal
    CAUTION = auto()     # Reduced operation, monitoring increased
    WARNING = auto()     # Significant concern, limiting actions
    CRITICAL = auto()    # Immediate action required
    EMERGENCY = auto()   # Full stop, human intervention needed


class SafetyCategory(Enum):
    """Categories of safety concerns."""
    COLLISION = "collision"
    BOUNDARY = "boundary"
    BATTERY = "battery"
    THERMAL = "thermal"
    HUMAN_PROXIMITY = "human_proximity"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    WORKSPACE = "workspace"


@dataclass
class SafetyViolation:
    """
    A safety violation record.

    Attributes:
        category: Type of safety concern
        level: Severity level
        message: Description of the violation
        timestamp: When the violation occurred
        source: Component that detected the violation
        data: Additional context data
    """
    category: SafetyCategory
    level: SafetyLevel
    message: str
    timestamp: float
    source: str = "unknown"
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'category': self.category.value,
            'level': self.level.name,
            'message': self.message,
            'timestamp': self.timestamp,
            'source': self.source,
            'data': self.data,
        }


@dataclass
class SafetyConfig:
    """
    Safety configuration parameters.

    All distances in meters, times in seconds.
    """
    # Collision avoidance
    min_obstacle_distance: float = 0.3      # Emergency stop distance
    slowdown_distance: float = 0.8          # Start slowing down
    collision_check_rate: float = 20.0      # Hz

    # Workspace boundaries
    workspace_min_x: float = -10.0
    workspace_max_x: float = 10.0
    workspace_min_y: float = -10.0
    workspace_max_y: float = 10.0
    boundary_buffer: float = 0.5            # Soft boundary before hard limit

    # Speed limits
    max_linear_speed: float = 0.5           # m/s
    max_angular_speed: float = 1.0          # rad/s
    human_proximity_speed: float = 0.2      # m/s when human nearby

    # Battery thresholds
    battery_low: float = 20.0               # Percent
    battery_critical: float = 10.0          # Percent

    # Thermal thresholds (Celsius)
    motor_temp_warning: float = 60.0
    motor_temp_critical: float = 80.0

    # Human safety
    human_safety_radius: float = 1.0        # Maintain this distance
    human_detection_enabled: bool = True

    # Timeouts
    heartbeat_timeout: float = 5.0          # Seconds without heartbeat
    command_timeout: float = 2.0            # Max time without new command


class SafetyMonitor(Node):
    """
    Safety monitor node for the home assistant robot.

    Monitors all safety-critical parameters and enforces
    safety constraints on robot operations.
    """

    def __init__(self):
        super().__init__('safety_monitor')

        # Configuration
        self._load_parameters()
        self.config = SafetyConfig()

        # State
        self.safety_level = SafetyLevel.NORMAL
        self.active_violations: List[SafetyViolation] = []
        self.emergency_stop_active = False
        self.last_heartbeat = time.time()
        self.last_command_time = time.time()

        # Sensor data
        self.min_obstacle_distance = float('inf')
        self.battery_level = 100.0
        self.motor_temperatures: Dict[str, float] = {}
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_velocity = {'linear': 0.0, 'angular': 0.0}
        self.humans_detected: List[Dict] = []

        # Thread safety
        self._lock = threading.Lock()

        # Callback group
        self._cb_group = ReentrantCallbackGroup()

        # QoS for reliable safety communication
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # === Subscribers ===
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self._scan_callback, 10,
            callback_group=self._cb_group
        )
        self.battery_sub = self.create_subscription(
            BatteryState, '/battery_state',
            self._battery_callback, 10,
            callback_group=self._cb_group
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel',
            self._cmd_vel_callback, 10,
            callback_group=self._cb_group
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose',
            self._pose_callback, 10,
            callback_group=self._cb_group
        )
        self.heartbeat_sub = self.create_subscription(
            Bool, '/safety/heartbeat',
            self._heartbeat_callback, 10,
            callback_group=self._cb_group
        )
        self.human_sub = self.create_subscription(
            String, '/perception/humans',
            self._human_callback, 10,
            callback_group=self._cb_group
        )

        # === Publishers ===
        self.emergency_pub = self.create_publisher(
            Bool, '/safety/emergency_stop', safety_qos
        )
        self.status_pub = self.create_publisher(
            String, '/safety/status', 10
        )
        self.safe_vel_pub = self.create_publisher(
            Twist, '/cmd_vel_safe', 10
        )
        self.violation_pub = self.create_publisher(
            String, '/safety/violations', 10
        )

        # === Timers ===
        self.check_timer = self.create_timer(
            1.0 / self.config.collision_check_rate,
            self._safety_check,
            callback_group=self._cb_group
        )
        self.status_timer = self.create_timer(
            1.0, self._publish_status,
            callback_group=self._cb_group
        )
        self.watchdog_timer = self.create_timer(
            0.5, self._watchdog_check,
            callback_group=self._cb_group
        )

        self.get_logger().info('Safety monitor initialized')

    def _load_parameters(self):
        """Load ROS parameters."""
        self.declare_parameter('min_obstacle_distance', 0.3)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('battery_low', 20.0)
        self.declare_parameter('human_safety_radius', 1.0)

    # =========================================================================
    # Sensor Callbacks
    # =========================================================================

    def _scan_callback(self, msg: LaserScan):
        """Process laser scan for obstacle detection."""
        if not msg.ranges:
            return

        # Filter valid ranges
        valid_ranges = [
            r for r in msg.ranges
            if msg.range_min < r < msg.range_max and not math.isnan(r)
        ]

        if valid_ranges:
            self.min_obstacle_distance = min(valid_ranges)
        else:
            self.min_obstacle_distance = float('inf')

    def _battery_callback(self, msg: BatteryState):
        """Update battery level."""
        self.battery_level = msg.percentage * 100.0

    def _cmd_vel_callback(self, msg: Twist):
        """Track commanded velocity and apply safety limits."""
        self.last_command_time = time.time()

        # Store current command
        self.current_velocity = {
            'linear': msg.linear.x,
            'angular': msg.angular.z
        }

        # Apply safety limits and publish safe velocity
        safe_twist = self._apply_safety_limits(msg)
        self.safe_vel_pub.publish(safe_twist)

    def _pose_callback(self, msg: PoseStamped):
        """Update robot pose."""
        self.robot_pose = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'yaw': self._quaternion_to_yaw(msg.pose.orientation)
        }

    def _heartbeat_callback(self, msg: Bool):
        """Update heartbeat timestamp."""
        if msg.data:
            self.last_heartbeat = time.time()

    def _human_callback(self, msg: String):
        """Update detected humans."""
        try:
            self.humans_detected = json.loads(msg.data)
        except json.JSONDecodeError:
            self.humans_detected = []

    # =========================================================================
    # Safety Checks
    # =========================================================================

    def _safety_check(self):
        """Main safety check routine."""
        with self._lock:
            self.active_violations.clear()

            # Run all safety checks
            self._check_collision()
            self._check_boundary()
            self._check_battery()
            self._check_human_proximity()

            # Determine overall safety level
            self._update_safety_level()

            # Handle emergency if needed
            if self.safety_level == SafetyLevel.EMERGENCY:
                self._trigger_emergency_stop("Safety check triggered emergency")

            # Publish violations
            if self.active_violations:
                self._publish_violations()

    def _check_collision(self):
        """Check for collision risk."""
        if self.min_obstacle_distance < self.config.min_obstacle_distance:
            self._add_violation(
                SafetyCategory.COLLISION,
                SafetyLevel.EMERGENCY,
                f"Obstacle too close: {self.min_obstacle_distance:.2f}m",
                data={'distance': self.min_obstacle_distance}
            )
        elif self.min_obstacle_distance < self.config.slowdown_distance:
            self._add_violation(
                SafetyCategory.COLLISION,
                SafetyLevel.WARNING,
                f"Obstacle nearby: {self.min_obstacle_distance:.2f}m",
                data={'distance': self.min_obstacle_distance}
            )

    def _check_boundary(self):
        """Check workspace boundaries."""
        x, y = self.robot_pose['x'], self.robot_pose['y']

        # Check hard boundaries
        if (x < self.config.workspace_min_x or x > self.config.workspace_max_x or
            y < self.config.workspace_min_y or y > self.config.workspace_max_y):
            self._add_violation(
                SafetyCategory.BOUNDARY,
                SafetyLevel.CRITICAL,
                f"Robot outside workspace at ({x:.2f}, {y:.2f})",
                data={'position': {'x': x, 'y': y}}
            )
            return

        # Check soft boundaries
        buffer = self.config.boundary_buffer
        if (x < self.config.workspace_min_x + buffer or
            x > self.config.workspace_max_x - buffer or
            y < self.config.workspace_min_y + buffer or
            y > self.config.workspace_max_y - buffer):
            self._add_violation(
                SafetyCategory.BOUNDARY,
                SafetyLevel.CAUTION,
                "Approaching workspace boundary",
                data={'position': {'x': x, 'y': y}}
            )

    def _check_battery(self):
        """Check battery level."""
        if self.battery_level < self.config.battery_critical:
            self._add_violation(
                SafetyCategory.BATTERY,
                SafetyLevel.CRITICAL,
                f"Critical battery: {self.battery_level:.0f}%",
                data={'level': self.battery_level}
            )
        elif self.battery_level < self.config.battery_low:
            self._add_violation(
                SafetyCategory.BATTERY,
                SafetyLevel.WARNING,
                f"Low battery: {self.battery_level:.0f}%",
                data={'level': self.battery_level}
            )

    def _check_human_proximity(self):
        """Check for humans nearby."""
        if not self.config.human_detection_enabled:
            return

        for human in self.humans_detected:
            pos = human.get('position', {})
            hx = pos.get('x', 0)
            hy = pos.get('y', 0)

            # Calculate distance to human
            rx, ry = self.robot_pose['x'], self.robot_pose['y']
            distance = math.sqrt((hx - rx)**2 + (hy - ry)**2)

            if distance < self.config.human_safety_radius:
                self._add_violation(
                    SafetyCategory.HUMAN_PROXIMITY,
                    SafetyLevel.WARNING,
                    f"Human within safety radius: {distance:.2f}m",
                    data={'distance': distance, 'human': human}
                )

    def _watchdog_check(self):
        """Check for system health issues."""
        now = time.time()

        # Check heartbeat
        if now - self.last_heartbeat > self.config.heartbeat_timeout:
            self._add_violation(
                SafetyCategory.SOFTWARE,
                SafetyLevel.CRITICAL,
                "Heartbeat timeout - system may be unresponsive",
                source="watchdog"
            )

        # Check command timeout (if robot should be moving)
        if (abs(self.current_velocity['linear']) > 0.01 or
            abs(self.current_velocity['angular']) > 0.01):
            if now - self.last_command_time > self.config.command_timeout:
                self.get_logger().warn("Command timeout - stopping robot")
                self._stop_robot()

    # =========================================================================
    # Safety Actions
    # =========================================================================

    def _apply_safety_limits(self, cmd: Twist) -> Twist:
        """Apply safety limits to velocity command."""
        safe_cmd = Twist()

        # Get speed limits based on current safety level
        max_linear = self.config.max_linear_speed
        max_angular = self.config.max_angular_speed

        # Reduce speed if warning
        if self.safety_level == SafetyLevel.WARNING:
            max_linear *= 0.5
            max_angular *= 0.5

        # Further reduce if human nearby
        if any(v.category == SafetyCategory.HUMAN_PROXIMITY
               for v in self.active_violations):
            max_linear = min(max_linear, self.config.human_proximity_speed)

        # Reduce based on obstacle distance
        if self.min_obstacle_distance < self.config.slowdown_distance:
            # Linear interpolation
            factor = (self.min_obstacle_distance - self.config.min_obstacle_distance) / \
                     (self.config.slowdown_distance - self.config.min_obstacle_distance)
            factor = max(0.0, min(1.0, factor))
            max_linear *= factor

        # Stop if emergency or critical
        if self.safety_level in (SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL):
            return safe_cmd  # Zero velocity

        # Apply limits
        safe_cmd.linear.x = max(-max_linear, min(max_linear, cmd.linear.x))
        safe_cmd.linear.y = max(-max_linear, min(max_linear, cmd.linear.y))
        safe_cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))

        return safe_cmd

    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.get_logger().error(f'EMERGENCY STOP: {reason}')

            # Publish emergency stop
            msg = Bool()
            msg.data = True
            self.emergency_pub.publish(msg)

            # Stop robot
            self._stop_robot()

    def _stop_robot(self):
        """Send stop command to robot."""
        stop_cmd = Twist()
        self.safe_vel_pub.publish(stop_cmd)

    def reset_emergency_stop(self):
        """Reset emergency stop (requires manual call)."""
        with self._lock:
            if not self.active_violations or \
               max(v.level.value for v in self.active_violations) < SafetyLevel.EMERGENCY.value:
                self.emergency_stop_active = False
                self.safety_level = SafetyLevel.NORMAL

                msg = Bool()
                msg.data = False
                self.emergency_pub.publish(msg)

                self.get_logger().info('Emergency stop cleared')
                return True
            else:
                self.get_logger().warn('Cannot clear emergency - violations still active')
                return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def _add_violation(
        self,
        category: SafetyCategory,
        level: SafetyLevel,
        message: str,
        source: str = "safety_monitor",
        data: Dict = None
    ):
        """Add a safety violation."""
        violation = SafetyViolation(
            category=category,
            level=level,
            message=message,
            timestamp=time.time(),
            source=source,
            data=data or {}
        )
        self.active_violations.append(violation)

        # Log based on severity
        if level == SafetyLevel.EMERGENCY:
            self.get_logger().error(f'SAFETY: {message}')
        elif level == SafetyLevel.CRITICAL:
            self.get_logger().error(f'SAFETY: {message}')
        elif level == SafetyLevel.WARNING:
            self.get_logger().warn(f'SAFETY: {message}')
        else:
            self.get_logger().info(f'SAFETY: {message}')

    def _update_safety_level(self):
        """Update overall safety level based on violations."""
        if not self.active_violations:
            self.safety_level = SafetyLevel.NORMAL
            return

        # Get highest severity
        max_level = max(v.level for v in self.active_violations)
        self.safety_level = max_level

    def _publish_status(self):
        """Publish safety status."""
        status = {
            'level': self.safety_level.name,
            'emergency_stop': self.emergency_stop_active,
            'battery': self.battery_level,
            'min_obstacle': self.min_obstacle_distance,
            'violation_count': len(self.active_violations),
            'timestamp': time.time(),
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _publish_violations(self):
        """Publish active violations."""
        violations = [v.to_dict() for v in self.active_violations]

        msg = String()
        msg.data = json.dumps(violations)
        self.violation_pub.publish(msg)

    @staticmethod
    def _quaternion_to_yaw(q) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def is_safe_to_move(self) -> bool:
        """Check if it's safe to move."""
        return (
            not self.emergency_stop_active and
            self.safety_level not in (SafetyLevel.EMERGENCY, SafetyLevel.CRITICAL)
        )

    def get_max_safe_speed(self) -> Tuple[float, float]:
        """Get current maximum safe speeds."""
        if not self.is_safe_to_move():
            return 0.0, 0.0

        max_linear = self.config.max_linear_speed
        max_angular = self.config.max_angular_speed

        if self.safety_level == SafetyLevel.WARNING:
            max_linear *= 0.5
            max_angular *= 0.5

        return max_linear, max_angular

    def get_safety_status(self) -> Dict:
        """Get current safety status."""
        return {
            'level': self.safety_level.name,
            'emergency_stop': self.emergency_stop_active,
            'safe_to_move': self.is_safe_to_move(),
            'max_speeds': self.get_max_safe_speed(),
            'violations': [v.to_dict() for v in self.active_violations],
        }


class SafetyChecker:
    """
    Standalone safety checker for action validation.

    Use this for pre-checking actions before execution.
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()

    def check_navigation_target(
        self,
        target: Dict[str, float],
        current_pose: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if navigation target is safe.

        Returns:
            Tuple of (is_safe, reason)
        """
        x = target.get('x', 0)
        y = target.get('y', 0)

        # Check workspace boundaries
        if (x < self.config.workspace_min_x or x > self.config.workspace_max_x or
            y < self.config.workspace_min_y or y > self.config.workspace_max_y):
            return False, f"Target ({x:.2f}, {y:.2f}) is outside workspace"

        return True, "Target is within safe workspace"

    def check_velocity(self, linear: float, angular: float) -> Tuple[bool, str]:
        """
        Check if velocity command is within limits.

        Returns:
            Tuple of (is_safe, reason)
        """
        if abs(linear) > self.config.max_linear_speed:
            return False, f"Linear velocity {linear:.2f} exceeds limit"

        if abs(angular) > self.config.max_angular_speed:
            return False, f"Angular velocity {angular:.2f} exceeds limit"

        return True, "Velocity within limits"

    def check_manipulation_target(
        self,
        position: List[float],
        workspace: Dict = None
    ) -> Tuple[bool, str]:
        """
        Check if manipulation target is reachable and safe.

        Returns:
            Tuple of (is_safe, reason)
        """
        if len(position) < 3:
            return False, "Invalid position format"

        x, y, z = position[:3]

        # Check height
        if z < 0:
            return False, "Target below ground plane"

        if z > 1.5:  # Max reach height
            return False, "Target too high"

        # Check reach (simplified)
        reach = math.sqrt(x**2 + y**2)
        if reach > 0.8:  # Max reach radius
            return False, f"Target too far: {reach:.2f}m"

        if reach < 0.1:
            return False, "Target too close to base"

        return True, "Target reachable"


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = SafetyMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
