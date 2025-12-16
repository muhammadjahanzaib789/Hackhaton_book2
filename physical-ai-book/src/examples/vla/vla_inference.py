#!/usr/bin/env python3
"""
VLA Inference Pipeline
Physical AI Book - Chapter 7: Vision-Language-Action

Optimized inference pipeline for real-time VLA deployment.
Includes image preprocessing, temporal buffering, and
action post-processing for robot execution.

Features:
- Temporal image buffering for context
- Efficient preprocessing pipeline
- Action smoothing and filtering
- Inference timing metrics

Usage:
    from physical_ai_examples.vla import VLAInference

    inference = VLAInference('model.pt', device='cuda')
    action = inference.predict(image, instruction)

Author: Physical AI Book
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Efficient image preprocessing for VLA inference.

    Handles resizing, normalization, and tensor conversion.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        device: str = 'cuda'
    ):
        self.target_size = target_size
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device = torch.device(device)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: [H, W, 3] RGB image (uint8 or float)

        Returns:
            tensor: [1, 3, H, W] normalized tensor
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        # Convert to PIL for reliable resizing
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))

        # Resize
        pil_image = pil_image.resize(self.target_size, Image.BILINEAR)

        # Convert to tensor
        tensor = torch.from_numpy(np.array(pil_image)).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        tensor = tensor / 255.0  # Normalize to [0, 1]

        # Move to device
        tensor = tensor.to(self.device)

        # Apply ImageNet normalization
        tensor = (tensor - self.mean) / self.std

        return tensor


class ActionSmoother:
    """
    Smooth action outputs for stable robot control.

    Uses exponential moving average to reduce jitter.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        action_dim: int = 7
    ):
        """
        Initialize smoother.

        Args:
            alpha: Smoothing factor (higher = more responsive)
            action_dim: Number of action dimensions
        """
        self.alpha = alpha
        self.action_dim = action_dim
        self.prev_action = None

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to action.

        Args:
            action: [action_dim] raw action

        Returns:
            smoothed: [action_dim] smoothed action
        """
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed.copy()
        return smoothed

    def reset(self):
        """Reset smoother state."""
        self.prev_action = None


class SimpleTokenizer:
    """
    Simple word-level tokenizer for instructions.

    For production, use a proper tokenizer (e.g., SentencePiece).
    """

    def __init__(self, max_length: int = 32):
        self.max_length = max_length

        # Simple vocabulary for common robot instructions
        self.vocab = {
            '<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3,
            'pick': 10, 'up': 11, 'put': 12, 'down': 13,
            'grab': 14, 'place': 15, 'move': 16, 'go': 17,
            'the': 20, 'a': 21, 'an': 22, 'to': 23, 'from': 24,
            'on': 25, 'in': 26, 'at': 27, 'near': 28,
            'red': 30, 'blue': 31, 'green': 32, 'yellow': 33,
            'black': 34, 'white': 35, 'orange': 36,
            'cup': 40, 'ball': 41, 'box': 42, 'block': 43,
            'bottle': 44, 'can': 45, 'bowl': 46, 'plate': 47,
            'table': 50, 'shelf': 51, 'desk': 52, 'floor': 53,
            'left': 60, 'right': 61, 'front': 62, 'back': 63,
            'bring': 70, 'fetch': 71, 'get': 72, 'take': 73,
            'open': 80, 'close': 81, 'push': 82, 'pull': 83,
        }

        self.id_to_word = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Instruction text

        Returns:
            tokens: List of token IDs
        """
        words = text.lower().split()
        tokens = [self.vocab.get('<start>')]

        for word in words:
            # Remove punctuation
            word = ''.join(c for c in word if c.isalnum())
            token_id = self.vocab.get(word, self.vocab['<unk>'])
            tokens.append(token_id)

        tokens.append(self.vocab.get('<end>'))

        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens += [self.vocab['<pad>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in tokens:
            if token_id in [0, 2, 3]:  # Skip special tokens
                continue
            words.append(self.id_to_word.get(token_id, '<unk>'))
        return ' '.join(words)


class VLAInference:
    """
    VLA inference pipeline for real-time robot control.

    Manages model loading, image buffering, preprocessing,
    and action post-processing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        buffer_size: int = 4,
        action_dim: int = 7,
        smooth_actions: bool = True,
        smooth_alpha: float = 0.7
    ):
        """
        Initialize VLA inference.

        Args:
            model_path: Path to model weights (optional)
            device: Device for inference
            buffer_size: Number of frames to buffer
            action_dim: Robot action dimension
            smooth_actions: Enable action smoothing
            smooth_alpha: Smoothing factor
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.buffer_size = buffer_size
        self.action_dim = action_dim

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Image buffer
        self.image_buffer = deque(maxlen=buffer_size)

        # Preprocessor and tokenizer
        self.preprocessor = ImagePreprocessor(device=str(self.device))
        self.tokenizer = SimpleTokenizer()

        # Action smoother
        self.smoother = ActionSmoother(smooth_alpha, action_dim) if smooth_actions else None

        # Cached instruction tokens
        self._cached_instruction = None
        self._cached_tokens = None

        # Metrics
        self._inference_times = deque(maxlen=100)

        logger.info(f"VLA Inference initialized on {self.device}")

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load VLA model."""
        from .simple_vla import SimpleVLA

        model = SimpleVLA(
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            action_dim=self.action_dim
        )

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")

        model.to(self.device)
        return model

    def set_instruction(self, instruction: str) -> None:
        """
        Set the current instruction.

        Caches tokenization for efficiency.

        Args:
            instruction: Natural language instruction
        """
        if instruction != self._cached_instruction:
            self._cached_instruction = instruction
            tokens = self.tokenizer.encode(instruction)
            self._cached_tokens = torch.tensor(
                tokens, device=self.device
            ).unsqueeze(0)
            logger.info(f"Instruction set: '{instruction}'")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict action from current observation.

        Args:
            image: [H, W, 3] RGB image
            instruction: Optional instruction (uses cached if not provided)

        Returns:
            action: [action_dim] action vector
        """
        start_time = time.time()

        # Update instruction if provided
        if instruction is not None:
            self.set_instruction(instruction)

        if self._cached_tokens is None:
            raise ValueError("No instruction set. Call set_instruction() first.")

        # Preprocess and buffer image
        image_tensor = self.preprocessor(image)
        self.image_buffer.append(image_tensor)

        # Pad buffer if needed
        while len(self.image_buffer) < self.buffer_size:
            self.image_buffer.appendleft(image_tensor)

        # Stack images [1, T, C, H, W]
        images = torch.cat(list(self.image_buffer), dim=0).unsqueeze(0)

        # Run inference
        action = self.model(images, self._cached_tokens)
        action = action.squeeze(0).cpu().numpy()

        # Apply smoothing
        if self.smoother:
            action = self.smoother.smooth(action)

        # Record timing
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)

        return action

    def reset(self) -> None:
        """Reset state for new episode."""
        self.image_buffer.clear()
        self._cached_instruction = None
        self._cached_tokens = None
        if self.smoother:
            self.smoother.reset()
        logger.debug("Inference state reset")

    def get_metrics(self) -> Dict[str, float]:
        """Get inference metrics."""
        if not self._inference_times:
            return {}

        times = list(self._inference_times)
        return {
            'mean_inference_time': np.mean(times),
            'max_inference_time': np.max(times),
            'min_inference_time': np.min(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
        }


class TensorRTVLAInference:
    """
    TensorRT-optimized VLA inference.

    Provides lower latency for production deployment.
    Requires TensorRT installation.
    """

    def __init__(
        self,
        engine_path: str,
        buffer_size: int = 4,
        action_dim: int = 7
    ):
        """
        Initialize TensorRT inference.

        Args:
            engine_path: Path to serialized TensorRT engine
            buffer_size: Image buffer size
            action_dim: Action dimension
        """
        self.buffer_size = buffer_size
        self.action_dim = action_dim

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT and PyCUDA required. Install with:\n"
                "pip install tensorrt pycuda"
            )

        # Load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(
                f.read()
            )
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        # Image buffer
        self.image_buffer = deque(maxlen=buffer_size)
        self.preprocessor = ImagePreprocessor(device='cpu')
        self.tokenizer = SimpleTokenizer()

        logger.info("TensorRT VLA inference initialized")

    def _allocate_buffers(self):
        """Allocate CUDA buffers."""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = abs(np.prod(shape))
            dtype = np.float32

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

    def predict(
        self,
        image: np.ndarray,
        instruction: str
    ) -> np.ndarray:
        """
        Predict action using TensorRT.

        Args:
            image: [H, W, 3] RGB image
            instruction: Natural language instruction

        Returns:
            action: [action_dim] action vector
        """
        import pycuda.driver as cuda

        # Preprocess
        image_tensor = self.preprocessor(image).cpu().numpy()
        self.image_buffer.append(image_tensor)

        while len(self.image_buffer) < self.buffer_size:
            self.image_buffer.appendleft(image_tensor)

        images = np.concatenate(list(self.image_buffer), axis=0)
        images = np.expand_dims(images, 0)  # Add batch dim

        tokens = np.array([self.tokenizer.encode(instruction)], dtype=np.int32)

        # Copy to device
        np.copyto(self.inputs[0]['host'], images.ravel())
        np.copyto(self.inputs[1]['host'], tokens.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod(inp['device'], inp['host'])

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])

        return self.outputs[0]['host'][:self.action_dim]


if __name__ == '__main__':
    # Demo
    print("VLA Inference Demo")
    print("=" * 50)

    # Create inference pipeline
    inference = VLAInference(device='cpu', buffer_size=4)

    # Set instruction
    inference.set_instruction("pick up the red cup")

    # Simulate inference loop
    print("\nRunning inference simulation...")
    for i in range(10):
        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Predict action
        action = inference.predict(image)

        print(f"Frame {i+1}: action = {action[:3]}... (truncated)")

    # Print metrics
    metrics = inference.get_metrics()
    print(f"\nMetrics:")
    print(f"  Mean inference time: {metrics['mean_inference_time']*1000:.2f} ms")
    print(f"  FPS: {metrics['fps']:.1f}")
