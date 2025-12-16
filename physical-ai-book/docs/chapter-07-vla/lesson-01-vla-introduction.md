---
sidebar_position: 1
title: "Lesson 1: VLA Fundamentals"
description: "Introduction to vision-language-action models for robotics"
---

# Vision-Language-Action Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the VLA paradigm and architecture
2. Compare VLAs with traditional robot control approaches
3. Identify appropriate use cases for VLA models
4. Understand key VLA model architectures

## Prerequisites

- Completed Chapters 1-6
- Understanding of transformers and LLMs
- Familiarity with computer vision basics

## What is Vision-Language-Action?

VLA models are end-to-end neural networks that directly map visual observations and language commands to robot actions.

```
┌─────────────────────────────────────────────────────────────┐
│              VLA Architecture Overview                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Traditional Approach:                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Vision  │→│Perception│→│ Planner │→│Controller│→ Action │
│  │ Input   │  │  Module │  │         │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       │           │             │            │              │
│  Language      Separate     Hand-coded    PID/MPC          │
│  (separate)    Models       Logic                          │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  VLA Approach:                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Vision-Language-Action Model            │   │
│  │  ┌─────────┐                         ┌─────────┐    │   │
│  │  │  Image  │─┐                    ┌─▶│ Action  │    │   │
│  │  └─────────┘ │  ┌───────────────┐ │  └─────────┘    │   │
│  │              ├─▶│  Transformer  │─┤                 │   │
│  │  ┌─────────┐ │  │    Backbone   │ │  ┌─────────┐    │   │
│  │  │Language │─┘  └───────────────┘ └─▶│  State  │    │   │
│  │  └─────────┘                         └─────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  End-to-End Learning                                        │
│  No hand-coded modules                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key VLA Advantages

| Aspect | Traditional | VLA |
|--------|-------------|-----|
| **Modularity** | Many separate modules | Single unified model |
| **Language** | Requires NLU pipeline | Native understanding |
| **Generalization** | Limited to programmed | Learns from data |
| **Development** | Manual engineering | Data collection |
| **Adaptation** | Requires reprogramming | Fine-tuning |

## VLA Model Architectures

### RT-1 (Robotics Transformer)

Google's foundational VLA model architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                    RT-1 Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image History (6 frames)         Language Instruction      │
│       │                                   │                 │
│       ▼                                   ▼                 │
│  ┌─────────────┐                  ┌─────────────┐          │
│  │ EfficientNet│                  │   USE       │          │
│  │  Encoder    │                  │  Encoder    │          │
│  └─────────────┘                  └─────────────┘          │
│       │                                   │                 │
│       │  Image Tokens                     │  Text Tokens   │
│       │                                   │                 │
│       └───────────────┬───────────────────┘                │
│                       │                                     │
│                       ▼                                     │
│              ┌─────────────────┐                           │
│              │    TokenLearner │  (Compress to 8 tokens)   │
│              └─────────────────┘                           │
│                       │                                     │
│                       ▼                                     │
│              ┌─────────────────┐                           │
│              │   Transformer   │  (8 layers)               │
│              │    Decoder      │                           │
│              └─────────────────┘                           │
│                       │                                     │
│                       ▼                                     │
│              ┌─────────────────┐                           │
│              │  Action Tokens  │  (Discretized)            │
│              │  [dx, dy, dz,   │                           │
│              │   roll, pitch,  │                           │
│              │   yaw, gripper] │                           │
│              └─────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### RT-2 (Vision-Language-Action)

Extends vision-language models to output actions.

```
┌─────────────────────────────────────────────────────────────┐
│                    RT-2 Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image           Language: "Pick up the apple"              │
│    │                        │                               │
│    │                        │                               │
│    ▼                        ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Pre-trained VLM (PaLI-X / PaLM-E)          │   │
│  │                                                     │   │
│  │  "The robot should pick up the apple by moving"    │   │
│  │  [dx: 256] [dy: 128] [dz: 100] [gripper: close]   │   │
│  │                                                     │   │
│  │  Actions encoded as special text tokens             │   │
│  └─────────────────────────────────────────────────────┘   │
│                       │                                     │
│                       │  Decode action tokens               │
│                       ▼                                     │
│              ┌─────────────────┐                           │
│              │   Action        │                           │
│              │   [0.1, -0.05,  │                           │
│              │    0.02, close] │                           │
│              └─────────────────┘                           │
│                                                             │
│  Key Innovation: Actions are just another "language"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### OpenVLA

Open-source VLA model architecture.

```python
"""
OpenVLA-style Architecture
Physical AI Book - Chapter 7

Simplified implementation of VLA concepts.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class VisionEncoder(nn.Module):
    """Encode images to feature vectors."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        # Use pretrained vision model
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Project to hidden dim
        self.project = nn.Linear(2048, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, T, C, H, W] batch of image sequences

        Returns:
            features: [B, T*N, D] flattened spatial features
        """
        B, T, C, H, W = images.shape

        # Process each frame
        images = images.view(B * T, C, H, W)
        features = self.backbone(images)  # [B*T, 2048, h, w]

        # Flatten spatial dimensions
        features = features.flatten(2).permute(0, 2, 1)  # [B*T, h*w, 2048]
        features = self.project(features)  # [B*T, h*w, D]

        # Reshape to batch
        features = features.view(B, -1, features.shape[-1])  # [B, T*h*w, D]

        return features


class LanguageEncoder(nn.Module):
    """Encode language instructions."""

    def __init__(self, hidden_dim: int = 768, vocab_size: int = 32000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_enc = nn.Embedding(512, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] token indices

        Returns:
            features: [B, L, D]
        """
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding(tokens) + self.position_enc(positions)


class ActionDecoder(nn.Module):
    """Decode actions from features."""

    def __init__(self, hidden_dim: int = 768, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim

        # Action bins for discretization
        self.num_bins = 256

        # Output heads for each action dimension
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_bins)
            for _ in range(action_dim)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] aggregated features

        Returns:
            actions: [B, action_dim] continuous actions
        """
        # Get logits for each action dimension
        logits = [head(features) for head in self.heads]

        # Convert to continuous actions
        actions = []
        for i, logit in enumerate(logits):
            # Softmax to get probabilities
            probs = torch.softmax(logit, dim=-1)

            # Expected value (soft argmax)
            bins = torch.arange(self.num_bins, device=logit.device).float()
            bins = (bins / self.num_bins) * 2 - 1  # Scale to [-1, 1]
            action = (probs * bins).sum(dim=-1)

            actions.append(action)

        return torch.stack(actions, dim=-1)


class SimpleVLA(nn.Module):
    """
    Simplified VLA model.

    Demonstrates core VLA concepts:
    - Multi-modal input (vision + language)
    - Transformer fusion
    - Action output
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        action_dim: int = 7
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Encoders
        self.vision_encoder = VisionEncoder(hidden_dim)
        self.language_encoder = LanguageEncoder(hidden_dim)

        # Learnable action query token
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Action decoder
        self.action_decoder = ActionDecoder(hidden_dim, action_dim)

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: [B, T, C, H, W] image sequence
            language_tokens: [B, L] language token indices

        Returns:
            actions: [B, action_dim] predicted actions
        """
        B = images.shape[0]

        # Encode inputs
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(language_tokens)

        # Expand action query for batch
        action_query = self.action_query.expand(B, -1, -1)

        # Concatenate all tokens
        # [action_query | vision | language]
        tokens = torch.cat([
            action_query,
            vision_features,
            language_features
        ], dim=1)

        # Apply transformer
        output = self.transformer(tokens)

        # Extract action query output
        action_features = output[:, 0, :]

        # Decode actions
        actions = self.action_decoder(action_features)

        return actions


# Example usage
def demo():
    model = SimpleVLA(
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        action_dim=7
    )

    # Dummy inputs
    images = torch.randn(2, 4, 3, 224, 224)  # 2 batches, 4 frames
    tokens = torch.randint(0, 1000, (2, 20))  # 2 batches, 20 tokens

    actions = model(images, tokens)
    print(f"Actions shape: {actions.shape}")  # [2, 7]
    print(f"Actions: {actions}")
```

## Training VLA Models

### Data Collection

```
┌─────────────────────────────────────────────────────────────┐
│              VLA Training Data Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Teleoperation Data Collection                           │
│     ┌─────────────────────────────────────────────────┐    │
│     │  Human demonstrates tasks via teleoperation     │    │
│     │  Record: Images, Actions, Language descriptions │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
│  2. Data Format                                             │
│     {                                                       │
│       "episode_id": "001",                                 │
│       "instruction": "pick up the red block",              │
│       "steps": [                                           │
│         {                                                  │
│           "image": <rgb_image>,                            │
│           "action": [dx, dy, dz, rx, ry, rz, gripper],    │
│           "state": [x, y, z, qx, qy, qz, qw, gripper]     │
│         },                                                 │
│         ...                                                │
│       ]                                                    │
│     }                                                       │
│                                                             │
│  3. Training Loop                                           │
│     - Sample episode and timestep                          │
│     - Forward pass with image + instruction                │
│     - Compute action prediction loss                       │
│     - Backpropagate and update                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Training Code

```python
"""
VLA Training Loop
Physical AI Book - Chapter 7
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import json
import os


class RobotDataset(Dataset):
    """Dataset for VLA training."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load episode metadata
        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List[Dict]:
        """Load all episode metadata."""
        episodes = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename)) as f:
                    episodes.append(json.load(f))
        return episodes

    def __len__(self):
        return sum(len(ep['steps']) for ep in self.episodes)

    def __getitem__(self, idx):
        # Find episode and step
        cumsum = 0
        for episode in self.episodes:
            if idx < cumsum + len(episode['steps']):
                step_idx = idx - cumsum
                break
            cumsum += len(episode['steps'])

        step = episode['steps'][step_idx]

        # Load image (would load from file in practice)
        image = torch.randn(3, 224, 224)  # Placeholder

        # Get action
        action = torch.tensor(step['action'], dtype=torch.float32)

        # Tokenize instruction (simplified)
        instruction = episode['instruction']
        tokens = torch.zeros(20, dtype=torch.long)  # Placeholder tokenization

        return {
            'image': image,
            'tokens': tokens,
            'action': action,
            'instruction': instruction
        }


def train_vla(model, train_loader, optimizer, epochs: int = 100):
    """Train VLA model."""
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            images = batch['image'].unsqueeze(1)  # Add time dim
            tokens = batch['tokens']
            target_actions = batch['action']

            pred_actions = model(images, tokens)

            # MSE loss on actions
            loss = torch.nn.functional.mse_loss(pred_actions, target_actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

## Deployment Considerations

### Inference Speed

```python
class VLAInference:
    """Optimized VLA inference for real-time control."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)

        # Load model
        self.model = SimpleVLA()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        # Image buffer for temporal context
        self.image_buffer = []
        self.buffer_size = 4

        # Pre-allocate tensors
        self.image_tensor = torch.zeros(
            1, self.buffer_size, 3, 224, 224,
            device=self.device
        )

    @torch.no_grad()
    def predict(self, image, instruction_tokens) -> torch.Tensor:
        """
        Predict action from current observation.

        Args:
            image: Current RGB image [H, W, 3]
            instruction_tokens: Tokenized instruction

        Returns:
            action: [7] action vector
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)

        # Update buffer
        self._update_buffer(image_tensor)

        # Run inference
        tokens = torch.tensor(
            instruction_tokens, device=self.device
        ).unsqueeze(0)

        action = self.model(self.image_tensor, tokens)

        return action.squeeze(0).cpu().numpy()

    def _preprocess_image(self, image):
        """Preprocess image for model."""
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        return transform(image).to(self.device)

    def _update_buffer(self, image):
        """Update temporal image buffer."""
        self.image_buffer.append(image)
        if len(self.image_buffer) > self.buffer_size:
            self.image_buffer.pop(0)

        # Pad if buffer not full
        while len(self.image_buffer) < self.buffer_size:
            self.image_buffer.insert(0, image)

        # Stack into tensor
        self.image_tensor[0] = torch.stack(self.image_buffer)
```

## Summary

Key takeaways from this lesson:

1. **VLAs** provide end-to-end learning for robot control
2. **Multi-modal fusion** combines vision and language
3. **Action discretization** enables autoregressive generation
4. **Large-scale data** is required for training
5. **Real-time inference** needs optimization

## Next Steps

In the [next lesson](./lesson-02-vla-deployment.md), we will:
- Deploy VLA models with ROS 2
- Optimize inference for real-time control
- Implement safety constraints

## Additional Resources

- [RT-1 Paper](https://arxiv.org/abs/2212.06817)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [OpenVLA](https://openvla.github.io/)
- [DROID Dataset](https://droid-dataset.github.io/)
