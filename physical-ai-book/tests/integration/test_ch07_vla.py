#!/usr/bin/env python3
"""
Integration Tests for Chapter 7: Vision-Language-Action
Physical AI Book

Tests the VLA pipeline including:
- Model architecture
- Forward pass
- Inference pipeline
- Training utilities
- Action processing

These tests verify VLA components work correctly together.

Usage:
    pytest tests/integration/test_ch07_vla.py -v

Author: Physical AI Book
License: MIT
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
from unittest.mock import MagicMock, patch
import tempfile
import json
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/examples'))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


@pytest.fixture
def small_vla(device):
    """Create a small VLA model for testing."""
    from vla.simple_vla import create_vla_model
    model = create_vla_model('small', action_dim=7)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def dummy_images(device):
    """Create dummy image batch."""
    # [batch_size, time_steps, channels, height, width]
    return torch.randn(2, 4, 3, 224, 224, device=device)


@pytest.fixture
def dummy_tokens(device):
    """Create dummy token batch."""
    # [batch_size, sequence_length]
    return torch.randint(0, 1000, (2, 20), device=device)


@pytest.fixture
def demo_data_dir(tmp_path):
    """Create temporary demo data directory."""
    # Create episode files
    for i in range(3):
        episode = {
            'episode_id': f'{i:04d}',
            'instruction': f'pick up object {i}',
            'steps': [
                {
                    'image_path': f'images/{i:04d}_{j:04d}.jpg',
                    'action': [0.1 * j for _ in range(7)],
                    'state': [0.0] * 8
                }
                for j in range(10)
            ]
        }
        with open(tmp_path / f'episode_{i:04d}.json', 'w') as f:
            json.dump(episode, f)

    return str(tmp_path)


# =============================================================================
# VLA Model Tests
# =============================================================================

class TestSimpleVLA:
    """Test SimpleVLA model architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        from vla.simple_vla import SimpleVLA

        model = SimpleVLA(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            action_dim=7
        )

        assert model is not None
        assert model.action_dim == 7
        assert model.hidden_dim == 256

    def test_model_factory(self):
        """Test model factory function."""
        from vla.simple_vla import create_vla_model

        # Test different sizes
        for size in ['small', 'medium', 'large']:
            model = create_vla_model(size, action_dim=7)
            assert model is not None

        # Test invalid size
        with pytest.raises(ValueError):
            create_vla_model('invalid_size')

    def test_forward_pass(self, small_vla, dummy_images, dummy_tokens, device):
        """Test forward pass produces correct output shape."""
        with torch.no_grad():
            actions = small_vla(dummy_images, dummy_tokens)

        assert actions.shape == (2, 7)  # [batch_size, action_dim]
        assert not torch.isnan(actions).any()

    def test_single_image_input(self, small_vla, device):
        """Test model handles single images (not sequences)."""
        single_image = torch.randn(2, 3, 224, 224, device=device)
        tokens = torch.randint(0, 1000, (2, 20), device=device)

        with torch.no_grad():
            actions = small_vla(single_image, tokens)

        assert actions.shape == (2, 7)

    def test_parameter_count(self, small_vla):
        """Test parameter counting methods."""
        total_params = small_vla.get_num_params()
        trainable_params = small_vla.get_num_trainable_params()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


class TestVisionEncoder:
    """Test vision encoder component."""

    def test_encoder_output_shape(self, device):
        """Test vision encoder output shape."""
        from vla.simple_vla import VisionEncoder

        encoder = VisionEncoder(hidden_dim=256).to(device)

        # Test with image sequence
        images = torch.randn(2, 4, 3, 224, 224, device=device)
        with torch.no_grad():
            features = encoder(images)

        # Output should be [batch, num_tokens, hidden_dim]
        assert features.dim() == 3
        assert features.shape[0] == 2
        assert features.shape[2] == 256

    def test_encoder_with_frozen_backbone(self, device):
        """Test encoder with frozen backbone."""
        from vla.simple_vla import VisionEncoder

        encoder = VisionEncoder(
            hidden_dim=256,
            freeze_backbone=True
        ).to(device)

        # Check backbone is frozen
        for param in encoder.backbone.parameters():
            assert not param.requires_grad


class TestLanguageEncoder:
    """Test language encoder component."""

    def test_encoder_output_shape(self, device):
        """Test language encoder output shape."""
        from vla.simple_vla import LanguageEncoder

        encoder = LanguageEncoder(
            hidden_dim=256,
            vocab_size=1000
        ).to(device)

        tokens = torch.randint(0, 1000, (2, 20), device=device)
        with torch.no_grad():
            features = encoder(tokens)

        assert features.shape == (2, 20, 256)

    def test_encoder_with_mask(self, device):
        """Test encoder with attention mask."""
        from vla.simple_vla import LanguageEncoder

        encoder = LanguageEncoder(hidden_dim=256).to(device)

        tokens = torch.randint(0, 1000, (2, 20), device=device)
        mask = torch.ones(2, 20, device=device)
        mask[:, 10:] = 0  # Mask second half

        with torch.no_grad():
            features = encoder(tokens, attention_mask=mask)

        assert features.shape == (2, 20, 256)


class TestActionDecoder:
    """Test action decoder component."""

    def test_decoder_output_shape(self, device):
        """Test decoder output shape."""
        from vla.simple_vla import ActionDecoder

        decoder = ActionDecoder(
            hidden_dim=256,
            action_dim=7,
            num_bins=256
        ).to(device)

        features = torch.randn(2, 256, device=device)
        with torch.no_grad():
            actions = decoder(features)

        assert actions.shape == (2, 7)

    def test_decoder_action_range(self, device):
        """Test decoder outputs in expected range."""
        from vla.simple_vla import ActionDecoder

        decoder = ActionDecoder(
            hidden_dim=256,
            action_dim=7,
            action_ranges=[(-1, 1)] * 7
        ).to(device)

        features = torch.randn(10, 256, device=device)
        with torch.no_grad():
            actions = decoder(features)

        # Actions should be approximately in [-1, 1]
        assert actions.min() >= -1.5
        assert actions.max() <= 1.5


# =============================================================================
# VLA Inference Tests
# =============================================================================

class TestVLAInference:
    """Test VLA inference pipeline."""

    def test_inference_creation(self):
        """Test inference pipeline creation."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu', buffer_size=4)
        assert inference is not None

    def test_set_instruction(self):
        """Test instruction setting."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu')
        inference.set_instruction("pick up the red cup")

        assert inference._cached_instruction == "pick up the red cup"
        assert inference._cached_tokens is not None

    def test_predict(self):
        """Test prediction with image."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu', buffer_size=4)
        inference.set_instruction("pick up the cup")

        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        action = inference.predict(image)

        assert action.shape == (7,)
        assert not np.isnan(action).any()

    def test_predict_with_instruction(self):
        """Test prediction with inline instruction."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu')
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        action = inference.predict(image, instruction="grab the ball")

        assert action.shape == (7,)

    def test_image_buffer(self):
        """Test image buffering."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu', buffer_size=4)
        inference.set_instruction("test")

        # Send multiple images
        for _ in range(6):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            inference.predict(image)

        # Buffer should be at max size
        assert len(inference.image_buffer) == 4

    def test_reset(self):
        """Test inference reset."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu')
        inference.set_instruction("test")

        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        inference.predict(image)

        inference.reset()

        assert len(inference.image_buffer) == 0
        assert inference._cached_instruction is None

    def test_metrics(self):
        """Test inference metrics collection."""
        from vla.vla_inference import VLAInference

        inference = VLAInference(device='cpu')
        inference.set_instruction("test")

        for _ in range(5):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            inference.predict(image)

        metrics = inference.get_metrics()

        assert 'mean_inference_time' in metrics
        assert 'fps' in metrics
        assert metrics['fps'] > 0


class TestActionSmoother:
    """Test action smoothing."""

    def test_smoother_basic(self):
        """Test basic smoothing."""
        from vla.vla_inference import ActionSmoother

        smoother = ActionSmoother(alpha=0.5, action_dim=7)

        action1 = np.array([1.0] * 7)
        action2 = np.array([0.0] * 7)

        smoothed1 = smoother.smooth(action1)
        smoothed2 = smoother.smooth(action2)

        # First action should be unchanged
        np.testing.assert_array_equal(smoothed1, action1)

        # Second should be blended
        expected = 0.5 * action2 + 0.5 * action1
        np.testing.assert_array_almost_equal(smoothed2, expected)

    def test_smoother_reset(self):
        """Test smoother reset."""
        from vla.vla_inference import ActionSmoother

        smoother = ActionSmoother(alpha=0.5)

        action = np.array([1.0] * 7)
        smoother.smooth(action)

        smoother.reset()
        assert smoother.prev_action is None


class TestSimpleTokenizer:
    """Test simple tokenizer."""

    def test_tokenizer_encode(self):
        """Test encoding."""
        from vla.vla_inference import SimpleTokenizer

        tokenizer = SimpleTokenizer(max_length=20)
        tokens = tokenizer.encode("pick up the red cup")

        assert len(tokens) == 20
        assert tokens[0] == 2  # <start>
        assert 0 in tokens  # Should have padding

    def test_tokenizer_unknown(self):
        """Test unknown token handling."""
        from vla.vla_inference import SimpleTokenizer

        tokenizer = SimpleTokenizer()
        tokens = tokenizer.encode("xyzzy foobar unknown")

        # Unknown words should map to <unk> (1)
        assert 1 in tokens


# =============================================================================
# VLA Training Tests
# =============================================================================

class TestRobotDemoDataset:
    """Test robot demonstration dataset."""

    def test_dataset_loading(self, demo_data_dir):
        """Test dataset loads correctly."""
        from vla.vla_training import RobotDemoDataset

        dataset = RobotDemoDataset(demo_data_dir)

        assert len(dataset) > 0

    def test_dataset_getitem(self, demo_data_dir):
        """Test dataset item retrieval."""
        from vla.vla_training import RobotDemoDataset

        dataset = RobotDemoDataset(demo_data_dir)
        sample = dataset[0]

        assert 'images' in sample
        assert 'tokens' in sample
        assert 'action' in sample
        assert sample['images'].dim() == 4  # [T, C, H, W]
        assert sample['tokens'].dim() == 1
        assert sample['action'].dim() == 1


class TestActionLoss:
    """Test action loss function."""

    def test_loss_computation(self, device):
        """Test loss computation."""
        from vla.vla_training import ActionLoss

        criterion = ActionLoss(action_dim=7)

        pred = torch.randn(4, 7, device=device)
        target = torch.randn(4, 7, device=device)

        losses = criterion(pred, target)

        assert 'total' in losses
        assert 'position' in losses
        assert 'rotation' in losses
        assert 'gripper' in losses
        assert losses['total'].item() > 0


class TestVLATrainer:
    """Test VLA trainer."""

    def test_trainer_creation(self, small_vla, demo_data_dir):
        """Test trainer creation."""
        from vla.vla_training import VLATrainer, RobotDemoDataset, TrainingConfig
        from torch.utils.data import DataLoader

        dataset = RobotDemoDataset(demo_data_dir)
        loader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(device='cpu', epochs=1)
        trainer = VLATrainer(small_vla, loader, config=config)

        assert trainer is not None

    def test_trainer_single_epoch(self, small_vla, demo_data_dir, tmp_path):
        """Test training for one epoch."""
        from vla.vla_training import VLATrainer, RobotDemoDataset, TrainingConfig
        from torch.utils.data import DataLoader

        dataset = RobotDemoDataset(demo_data_dir)
        loader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(
            device='cpu',
            epochs=1,
            mixed_precision=False,
            log_every=100  # Suppress logging
        )
        trainer = VLATrainer(small_vla, loader, config=config)

        # Should not raise
        trainer.train(epochs=1, save_dir=str(tmp_path))

        # Check checkpoint was saved
        assert (tmp_path / 'final_model.pt').exists()


# =============================================================================
# Safety Filter Tests
# =============================================================================

class TestSafetyFilter:
    """Test VLA safety filter."""

    def test_velocity_limiting(self):
        """Test velocity limiting."""
        from vla.vla_ros_node import SafetyFilter

        filter = SafetyFilter(
            max_linear_vel=0.1,
            max_angular_vel=0.3
        )

        # Action with excessive velocity
        action = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5])

        filtered, is_safe, violations = filter.filter(action)

        assert not is_safe
        assert len(violations) > 0

        # Linear velocity should be clamped
        linear_mag = np.linalg.norm(filtered[:3])
        assert linear_mag <= 0.1 + 1e-6

    def test_gripper_saturation(self):
        """Test gripper value saturation."""
        from vla.vla_ros_node import SafetyFilter

        filter = SafetyFilter()

        # Action with out-of-range gripper
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])

        filtered, _, _ = filter.filter(action)

        assert 0.0 <= filtered[6] <= 1.0

    def test_rate_limiting(self):
        """Test action rate limiting."""
        from vla.vla_ros_node import SafetyFilter

        filter = SafetyFilter(max_linear_vel=1.0)
        filter.max_delta = 0.1

        # First action
        action1 = np.array([0.0] * 7)
        filter.filter(action1)

        # Large change
        action2 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5])
        filtered, is_safe, violations = filter.filter(action2)

        # Should be rate limited
        delta = np.linalg.norm(filtered[:6] - action1[:6])
        assert delta <= 0.1 + 1e-6

    def test_filter_reset(self):
        """Test filter reset."""
        from vla.vla_ros_node import SafetyFilter

        filter = SafetyFilter()

        action = np.array([0.1] * 7)
        filter.filter(action)

        filter.reset()
        assert filter.prev_action is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestVLAIntegration:
    """End-to-end integration tests."""

    def test_full_inference_pipeline(self):
        """Test complete inference pipeline."""
        from vla.simple_vla import create_vla_model
        from vla.vla_inference import VLAInference

        # Create inference pipeline
        inference = VLAInference(device='cpu', buffer_size=4)

        # Set instruction
        inference.set_instruction("pick up the red cup")

        # Simulate control loop
        actions = []
        for _ in range(10):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            action = inference.predict(image)
            actions.append(action)

        # Should produce consistent outputs
        assert len(actions) == 10
        assert all(a.shape == (7,) for a in actions)

    def test_training_to_inference(self, demo_data_dir, tmp_path):
        """Test training then inference."""
        from vla.simple_vla import create_vla_model
        from vla.vla_training import VLATrainer, RobotDemoDataset, TrainingConfig
        from vla.vla_inference import VLAInference
        from torch.utils.data import DataLoader

        # Train model
        model = create_vla_model('small')
        dataset = RobotDemoDataset(demo_data_dir)
        loader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(
            device='cpu',
            epochs=1,
            mixed_precision=False,
            log_every=100
        )
        trainer = VLATrainer(model, loader, config=config)
        trainer.train(epochs=1, save_dir=str(tmp_path))

        # Load in inference
        model_path = tmp_path / 'final_model.pt'
        inference = VLAInference(
            model_path=str(model_path),
            device='cpu'
        )

        # Run inference
        inference.set_instruction("test instruction")
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        action = inference.predict(image)

        assert action.shape == (7,)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
