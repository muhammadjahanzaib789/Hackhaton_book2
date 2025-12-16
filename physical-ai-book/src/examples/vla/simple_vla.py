#!/usr/bin/env python3
"""
Simple VLA Model Implementation
Physical AI Book - Chapter 7: Vision-Language-Action

A simplified VLA model demonstrating the core architecture:
- Vision encoder for image features
- Language encoder for instruction understanding
- Transformer fusion for multi-modal reasoning
- Action decoder for robot control output

This is an educational implementation - production VLAs use
larger backbones and more sophisticated architectures.

Usage:
    from physical_ai_examples.vla import SimpleVLA

    model = SimpleVLA(hidden_dim=512, action_dim=7)
    actions = model(images, language_tokens)

Author: Physical AI Book
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class VisionEncoder(nn.Module):
    """
    Vision encoder using a CNN backbone.

    Extracts spatial features from images and projects them
    to the transformer hidden dimension.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Use ResNet as backbone
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                resnet = resnet50(weights=None)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import resnet50
            resnet = resnet50(pretrained=pretrained)

        # Remove classification head, keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to hidden dimension
        self.project = nn.Sequential(
            nn.Conv2d(2048, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Learnable spatial position embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, hidden_dim, 7, 7) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature tokens.

        Args:
            images: [B, T, C, H, W] batch of image sequences
                   or [B, C, H, W] single images

        Returns:
            features: [B, N, D] flattened spatial features
                     N = T * h * w for sequences
        """
        # Handle both single images and sequences
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            is_sequence = True
        else:
            B = images.shape[0]
            T = 1
            is_sequence = False

        # Extract features through backbone
        features = self.backbone(images)  # [B*T, 2048, h, w]

        # Project to hidden dimension
        features = self.project(features)  # [B*T, D, h, w]

        # Add spatial position embeddings
        features = features + self.spatial_pos

        # Flatten spatial dimensions
        features = features.flatten(2).permute(0, 2, 1)  # [B*T, h*w, D]

        # Reshape for batch
        if is_sequence:
            features = features.view(B, -1, self.hidden_dim)  # [B, T*h*w, D]
        else:
            features = features.view(B, -1, self.hidden_dim)  # [B, h*w, D]

        return features


class LanguageEncoder(nn.Module):
    """
    Language encoder for instruction understanding.

    Uses learned embeddings with positional encoding.
    For production, would use a pretrained language model.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 32000,
        max_length: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_length, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode language tokens.

        Args:
            tokens: [B, L] token indices
            attention_mask: [B, L] mask for padding (1 = attend, 0 = ignore)

        Returns:
            features: [B, L, D] encoded features
        """
        # Embed tokens
        x = self.token_embedding(tokens)  # [B, L, D]

        # Add positional encoding (transpose for pos_encoding format)
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)  # [B, L, D]

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Project output
        x = self.output_proj(x)

        return x


class ActionDecoder(nn.Module):
    """
    Decode actions from fused features.

    Uses discretized action bins for more stable training,
    then converts to continuous actions.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 7,
        num_bins: int = 256,
        action_ranges: Optional[list] = None
    ):
        """
        Initialize action decoder.

        Args:
            hidden_dim: Input feature dimension
            action_dim: Number of action dimensions
            num_bins: Number of bins for discretization
            action_ranges: [(min, max), ...] for each action dimension
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins

        # Default action ranges (normalized)
        if action_ranges is None:
            action_ranges = [(-1.0, 1.0)] * action_dim
        self.action_ranges = action_ranges

        # Pre-MLP
        self.pre_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate head for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_bins)
            for _ in range(action_dim)
        ])

    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Decode actions from features.

        Args:
            features: [B, D] aggregated features
            temperature: Softmax temperature for sampling

        Returns:
            actions: [B, action_dim] continuous actions
        """
        # Pre-process features
        x = self.pre_mlp(features)

        actions = []
        for i, head in enumerate(self.action_heads):
            # Get logits for this action dimension
            logits = head(x) / temperature  # [B, num_bins]

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Create bin centers
            min_val, max_val = self.action_ranges[i]
            bins = torch.linspace(
                min_val, max_val, self.num_bins,
                device=features.device
            )

            # Compute expected value (soft argmax)
            action = (probs * bins).sum(dim=-1)  # [B]
            actions.append(action)

        return torch.stack(actions, dim=-1)  # [B, action_dim]

    def get_action_logits(self, features: torch.Tensor) -> list:
        """Get raw logits for each action dimension."""
        x = self.pre_mlp(features)
        return [head(x) for head in self.action_heads]


class SimpleVLA(nn.Module):
    """
    Simple Vision-Language-Action model.

    Demonstrates core VLA architecture:
    1. Encode visual observations
    2. Encode language instructions
    3. Fuse multi-modal features via transformer
    4. Decode robot actions

    This is a simplified educational model. Production VLAs
    use larger pre-trained backbones and more data.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        action_dim: int = 7,
        vocab_size: int = 32000,
        dropout: float = 0.1,
        freeze_vision: bool = False
    ):
        """
        Initialize VLA model.

        Args:
            hidden_dim: Transformer hidden dimension
            num_layers: Number of fusion transformer layers
            num_heads: Number of attention heads
            action_dim: Robot action dimension (e.g., 7 for 6-DOF + gripper)
            vocab_size: Language vocabulary size
            dropout: Dropout rate
            freeze_vision: Freeze vision backbone weights
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Encoders
        self.vision_encoder = VisionEncoder(
            hidden_dim=hidden_dim,
            pretrained=True,
            freeze_backbone=freeze_vision
        )
        self.language_encoder = LanguageEncoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout
        )

        # Learnable action query token
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Modality type embeddings
        self.modality_embedding = nn.Embedding(3, hidden_dim)  # vision, language, action

        # Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Action decoder
        self.action_decoder = ActionDecoder(
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: [B, T, C, H, W] image sequence or [B, C, H, W] single image
            language_tokens: [B, L] language token indices
            language_mask: [B, L] attention mask for language

        Returns:
            actions: [B, action_dim] predicted robot actions
        """
        B = images.shape[0]
        device = images.device

        # Encode inputs
        vision_features = self.vision_encoder(images)  # [B, N_v, D]
        language_features = self.language_encoder(
            language_tokens, language_mask
        )  # [B, N_l, D]

        # Add modality embeddings
        vision_type = torch.zeros(
            vision_features.shape[1], dtype=torch.long, device=device
        )
        language_type = torch.ones(
            language_features.shape[1], dtype=torch.long, device=device
        )
        action_type = torch.full((1,), 2, dtype=torch.long, device=device)

        vision_features = vision_features + self.modality_embedding(vision_type)
        language_features = language_features + self.modality_embedding(language_type)

        # Expand action query for batch
        action_query = self.action_query.expand(B, -1, -1)
        action_query = action_query + self.modality_embedding(action_type)

        # Concatenate all tokens: [action_query | vision | language]
        tokens = torch.cat([
            action_query,
            vision_features,
            language_features
        ], dim=1)

        # Apply fusion transformer
        fused = self.fusion_transformer(tokens)

        # Extract action query output (first token)
        action_features = fused[:, 0, :]  # [B, D]

        # Decode actions
        actions = self.action_decoder(action_features)

        return actions

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vla_model(
    size: str = 'small',
    action_dim: int = 7,
    **kwargs
) -> SimpleVLA:
    """
    Factory function to create VLA models of different sizes.

    Args:
        size: Model size ('small', 'medium', 'large')
        action_dim: Robot action dimension
        **kwargs: Additional arguments passed to SimpleVLA

    Returns:
        Configured SimpleVLA model
    """
    configs = {
        'small': {
            'hidden_dim': 384,
            'num_layers': 4,
            'num_heads': 6,
        },
        'medium': {
            'hidden_dim': 768,
            'num_layers': 6,
            'num_heads': 8,
        },
        'large': {
            'hidden_dim': 1024,
            'num_layers': 8,
            'num_heads': 16,
        }
    }

    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(configs.keys())}")

    config = configs[size]
    config.update(kwargs)
    config['action_dim'] = action_dim

    return SimpleVLA(**config)


if __name__ == '__main__':
    # Demo
    print("SimpleVLA Demo")
    print("=" * 50)

    model = create_vla_model('small', action_dim=7)
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    # Create dummy inputs
    images = torch.randn(2, 4, 3, 224, 224)  # 2 batches, 4 frames
    tokens = torch.randint(0, 1000, (2, 20))  # 2 batches, 20 tokens

    # Forward pass
    with torch.no_grad():
        actions = model(images, tokens)

    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Tokens: {tokens.shape}")
    print(f"\nOutput shape: {actions.shape}")
    print(f"Actions: {actions}")
