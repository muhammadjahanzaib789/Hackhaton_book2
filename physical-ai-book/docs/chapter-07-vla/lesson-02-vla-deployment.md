---
sidebar_position: 2
title: "Lesson 2: VLA Deployment"
description: "Deploying VLA models for real-time robot control"
---

# VLA Deployment

## Learning Objectives

By the end of this lesson, you will be able to:

1. Deploy VLA models in ROS 2 systems
2. Optimize inference for real-time control
3. Implement safety layers for VLA outputs
4. Handle deployment challenges

## Prerequisites

- Completed Lesson 1 (VLA Fundamentals)
- ROS 2 experience from previous chapters
- Understanding of GPU inference

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              VLA Deployment Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   VLA Controller Node                │   │
│  │                                                     │   │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐    │   │
│  │  │  Camera   │   │ Language  │   │   State   │    │   │
│  │  │  Input    │   │   Input   │   │   Input   │    │   │
│  │  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘    │   │
│  │        │               │               │          │   │
│  │        └───────────────┼───────────────┘          │   │
│  │                        │                          │   │
│  │                        ▼                          │   │
│  │               ┌─────────────────┐                 │   │
│  │               │   VLA Model     │                 │   │
│  │               │   (GPU/NPU)     │                 │   │
│  │               └────────┬────────┘                 │   │
│  │                        │                          │   │
│  │                        ▼                          │   │
│  │               ┌─────────────────┐                 │   │
│  │               │  Safety Filter  │                 │   │
│  │               └────────┬────────┘                 │   │
│  │                        │                          │   │
│  │                        ▼                          │   │
│  │               ┌─────────────────┐                 │   │
│  │               │ Action Output   │                 │   │
│  │               └─────────────────┘                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│       │                       │                            │
│       ▼                       ▼                            │
│  ┌─────────────┐       ┌─────────────┐                    │
│  │ Arm Control │       │ Navigation  │                    │
│  │  (MoveIt2)  │       │   (Nav2)    │                    │
│  └─────────────┘       └─────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ROS 2 VLA Node

```python
#!/usr/bin/env python3
"""
VLA Controller Node
Physical AI Book - Chapter 7

ROS 2 node for real-time VLA-based robot control.

Usage:
    ros2 run physical_ai_examples vla_controller

Expected Output:
    [INFO] [vla_controller]: Loading VLA model...
    [INFO] [vla_controller]: VLA controller ready
    [INFO] [vla_controller]: Instruction: "pick up the cup"
    [INFO] [vla_controller]: Action: [0.02, -0.01, 0.03, 0.0, 0.0, 0.0, 0.8]

Dependencies:
    - rclpy
    - torch
    - sensor_msgs
    - geometry_msgs
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
import torch
from typing import Optional, List
from collections import deque
import threading
import time


class VLAControllerNode(Node):
    """
    VLA-based robot controller.

    Runs VLA inference at fixed rate and outputs robot actions.
    """

    def __init__(self):
        super().__init__('vla_controller')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('image_history', 4)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('safety_enabled', True)

        model_path = self.get_parameter('model_path').value
        self.control_rate = self.get_parameter('control_rate').value
        self.image_history = self.get_parameter('image_history').value
        self.device = self.get_parameter('device').value
        self.safety_enabled = self.get_parameter('safety_enabled').value

        # Initialize model
        self.get_logger().info('Loading VLA model...')
        self.model = self._load_model(model_path)
        self.get_logger().info('VLA model loaded')

        # Image processing
        self.bridge = CvBridge()
        self.image_buffer = deque(maxlen=self.image_history)
        self.current_image = None

        # State
        self.current_instruction = None
        self.instruction_tokens = None
        self.is_active = False
        self.current_joint_state = None

        # Safety filter
        self.safety_filter = SafetyFilter() if self.safety_enabled else None

        # Callback group for parallel processing
        self._callback_group = ReentrantCallbackGroup()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self._image_callback, 10,
            callback_group=self._callback_group
        )
        self.instruction_sub = self.create_subscription(
            String, '/vla/instruction',
            self._instruction_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self._joint_callback, 10
        )

        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/vla/action', 10
        )
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray, '/arm_controller/command', 10
        )
        self.gripper_cmd_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/command', 10
        )
        self.status_pub = self.create_publisher(
            String, '/vla/status', 10
        )

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_loop,
            callback_group=self._callback_group
        )

        # Lock for thread safety
        self._lock = threading.Lock()

        self.get_logger().info('VLA controller ready')

    def _load_model(self, model_path: str):
        """Load VLA model."""
        # In production, load actual model
        # For demo, create simple model
        from .simple_vla import SimpleVLA

        model = SimpleVLA(
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            action_dim=7
        )

        if model_path:
            try:
                model.load_state_dict(torch.load(model_path))
            except Exception as e:
                self.get_logger().warn(f'Could not load weights: {e}')

        model.to(self.device)
        model.eval()

        return model

    def _image_callback(self, msg: Image):
        """Process incoming camera images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.current_image = cv_image

            with self._lock:
                self.image_buffer.append(cv_image)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def _instruction_callback(self, msg: String):
        """Handle new instruction."""
        instruction = msg.data.strip()

        if not instruction:
            self.is_active = False
            self.current_instruction = None
            self.get_logger().info('Instruction cleared, stopping')
            return

        self.current_instruction = instruction
        self.instruction_tokens = self._tokenize(instruction)
        self.is_active = True

        self.get_logger().info(f'Instruction: "{instruction}"')

    def _joint_callback(self, msg: JointState):
        """Update current joint state."""
        self.current_joint_state = msg

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize instruction text."""
        # Simple tokenization (would use proper tokenizer)
        # Convert to lowercase and split
        words = text.lower().split()

        # Simple vocabulary mapping
        vocab = {
            'pick': 1, 'up': 2, 'the': 3, 'cup': 4, 'red': 5,
            'blue': 6, 'put': 7, 'down': 8, 'move': 9, 'to': 10,
            'left': 11, 'right': 12, 'grab': 13, 'place': 14,
            'block': 15, 'ball': 16, 'box': 17
        }

        tokens = [vocab.get(w, 0) for w in words]
        tokens = tokens[:20]  # Max length
        tokens += [0] * (20 - len(tokens))  # Pad

        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def _control_loop(self):
        """Main control loop - runs at fixed rate."""
        if not self.is_active or self.instruction_tokens is None:
            return

        with self._lock:
            if len(self.image_buffer) < self.image_history:
                return

            # Prepare image input
            images = self._prepare_images()

        # Run inference
        action = self._infer(images, self.instruction_tokens)

        if action is None:
            return

        # Apply safety filter
        if self.safety_filter:
            action, is_safe = self.safety_filter.filter(
                action, self.current_joint_state
            )
            if not is_safe:
                self.get_logger().warn('Action modified by safety filter')

        # Publish action
        self._publish_action(action)

    def _prepare_images(self) -> torch.Tensor:
        """Prepare image tensor from buffer."""
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        images = []
        for img in self.image_buffer:
            img_tensor = transform(img)
            images.append(img_tensor)

        # Stack into [1, T, C, H, W]
        images = torch.stack(images).unsqueeze(0).to(self.device)

        return images

    @torch.no_grad()
    def _infer(self, images: torch.Tensor,
               tokens: torch.Tensor) -> Optional[np.ndarray]:
        """Run VLA inference."""
        try:
            action = self.model(images, tokens)
            return action.squeeze(0).cpu().numpy()

        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            return None

    def _publish_action(self, action: np.ndarray):
        """Publish action to robot controllers."""
        # Full action vector
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        # Split into arm and gripper commands
        # Assuming action = [dx, dy, dz, drx, dry, drz, gripper]
        arm_action = action[:6]  # End effector delta
        gripper_action = action[6:]  # Gripper state

        # Arm command (would convert to joint space in production)
        arm_msg = Float64MultiArray()
        arm_msg.data = arm_action.tolist()
        self.arm_cmd_pub.publish(arm_msg)

        # Gripper command
        gripper_msg = Float64MultiArray()
        gripper_msg.data = gripper_action.tolist()
        self.gripper_cmd_pub.publish(gripper_msg)

        self.get_logger().debug(f'Action: {action}')


class SafetyFilter:
    """
    Safety filter for VLA outputs.

    Applies constraints to ensure safe robot behavior.
    """

    def __init__(self):
        # Velocity limits
        self.max_linear_vel = 0.1  # m/step
        self.max_angular_vel = 0.2  # rad/step

        # Workspace limits
        self.workspace_min = np.array([-0.5, -0.5, 0.0])
        self.workspace_max = np.array([0.5, 0.5, 1.0])

        # Joint limits (example)
        self.joint_limits = [
            (-np.pi, np.pi),    # Joint 1
            (-np.pi/2, np.pi/2),
            (-np.pi, np.pi),
            (-np.pi, 0),
            (-np.pi, np.pi),
            (-np.pi/4, np.pi/4),
            (-np.pi/4, np.pi/4),
        ]

        # Force/torque limits
        self.max_force = 20.0  # N

    def filter(self, action: np.ndarray,
               joint_state=None) -> tuple:
        """
        Filter action through safety constraints.

        Args:
            action: Raw VLA action output
            joint_state: Current joint state (optional)

        Returns:
            (filtered_action, is_safe) tuple
        """
        is_safe = True
        filtered = action.copy()

        # Velocity limiting
        linear = filtered[:3]
        angular = filtered[3:6]

        linear_mag = np.linalg.norm(linear)
        if linear_mag > self.max_linear_vel:
            linear = linear / linear_mag * self.max_linear_vel
            filtered[:3] = linear
            is_safe = False

        angular_mag = np.linalg.norm(angular)
        if angular_mag > self.max_angular_vel:
            angular = angular / angular_mag * self.max_angular_vel
            filtered[3:6] = angular
            is_safe = False

        # Gripper saturation
        filtered[6] = np.clip(filtered[6], 0.0, 1.0)

        return filtered, is_safe


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = VLAControllerNode()

    # Demo: Send a test instruction
    import time
    time.sleep(2.0)

    instruction_msg = String()
    instruction_msg.data = "pick up the red cup"
    node._instruction_callback(instruction_msg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Inference Optimization

### TensorRT Deployment

```python
"""
TensorRT VLA Inference
Physical AI Book - Chapter 7

Optimized inference using TensorRT for NVIDIA GPUs.
"""

import tensorrt as trt
import numpy as np


class TRTVLAInference:
    """
    TensorRT-optimized VLA inference.

    Provides low-latency inference for real-time control.
    """

    def __init__(self, engine_path: str):
        """
        Load TensorRT engine.

        Args:
            engine_path: Path to serialized TRT engine
        """
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(
                f.read()
            )

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate input/output buffers."""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })

    def predict(self, images: np.ndarray,
                tokens: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Args:
            images: [B, T, C, H, W] image tensor
            tokens: [B, L] token tensor

        Returns:
            actions: [B, action_dim] predicted actions
        """
        import pycuda.driver as cuda

        # Copy inputs to device
        np.copyto(self.inputs[0]['host'], images.ravel())
        np.copyto(self.inputs[1]['host'], tokens.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod(inp['device'], inp['host'])

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])

        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])


def convert_to_tensorrt(
    model,
    image_shape: tuple,
    token_length: int,
    output_path: str,
    fp16: bool = True
):
    """
    Convert PyTorch model to TensorRT.

    Args:
        model: PyTorch model
        image_shape: (B, T, C, H, W) input shape
        token_length: Token sequence length
        output_path: Where to save engine
        fp16: Enable FP16 inference
    """
    import torch

    # Export to ONNX first
    dummy_images = torch.randn(image_shape)
    dummy_tokens = torch.randint(0, 1000, (image_shape[0], token_length))

    onnx_path = output_path.replace('.trt', '.onnx')

    torch.onnx.export(
        model,
        (dummy_images, dummy_tokens),
        onnx_path,
        input_names=['images', 'tokens'],
        output_names=['actions'],
        dynamic_axes={
            'images': {0: 'batch'},
            'tokens': {0: 'batch'},
            'actions': {0: 'batch'}
        }
    )

    # Convert ONNX to TensorRT
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt.Logger())

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f'TensorRT engine saved to {output_path}')
```

## Summary

Key takeaways from this lesson:

1. **ROS 2 integration** enables VLA in robotic systems
2. **Safety filtering** is critical for VLA outputs
3. **TensorRT** provides optimized GPU inference
4. **Control rate** must match robot capabilities
5. **Multi-modal synchronization** requires careful handling

## Next Steps

Continue to [Chapter 8: Capstone Project](../chapter-08-capstone/lesson-01-project-overview.md) to:
- Build a complete physical AI system
- Integrate all learned concepts
- Deploy an end-to-end application

## Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime for Robotics](https://onnxruntime.ai/)
- [ROS 2 Real-time](https://docs.ros.org/en/humble/Tutorials/Real-time.html)
