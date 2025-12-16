#!/usr/bin/env python3
"""
VLA Training Pipeline
Physical AI Book - Chapter 7: Vision-Language-Action

Training utilities for VLA models including:
- Dataset loading for robot demonstrations
- Training loop with logging
- Evaluation metrics
- Checkpoint management

Usage:
    from physical_ai_examples.vla import VLATrainer

    trainer = VLATrainer(model, train_loader, val_loader)
    trainer.train(epochs=100)

Author: Physical AI Book
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    eval_every: int = 5
    save_every: int = 10
    log_every: int = 10
    device: str = 'cuda'
    mixed_precision: bool = True


class RobotDemoDataset(Dataset):
    """
    Dataset for robot demonstration data.

    Expected data format:
    {
        "episode_id": "001",
        "instruction": "pick up the red cup",
        "steps": [
            {
                "image_path": "images/001_000.jpg",
                "action": [dx, dy, dz, drx, dry, drz, gripper],
                "state": [x, y, z, qx, qy, qz, qw, gripper_state]
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_length: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        history_length: int = 4
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing episode JSON files
            transform: Image transform
            max_length: Maximum instruction length
            image_size: Target image size
            history_length: Number of history frames
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_length = max_length
        self.image_size = image_size
        self.history_length = history_length

        # Load episode metadata
        self.episodes = self._load_episodes()
        self.samples = self._create_samples()

        # Simple tokenizer (would use proper one in production)
        self.vocab = self._build_vocab()

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.episodes)} episodes")

    def _load_episodes(self) -> List[Dict]:
        """Load all episode metadata."""
        episodes = []
        for json_file in self.data_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    episodes.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        return episodes

    def _create_samples(self) -> List[Tuple[int, int]]:
        """Create sample indices (episode_idx, step_idx)."""
        samples = []
        for ep_idx, episode in enumerate(self.episodes):
            num_steps = len(episode.get('steps', []))
            for step_idx in range(num_steps):
                samples.append((ep_idx, step_idx))
        return samples

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from episodes."""
        words = set()
        for episode in self.episodes:
            instruction = episode.get('instruction', '')
            words.update(instruction.lower().split())

        vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        for i, word in enumerate(sorted(words)):
            vocab[word] = i + 4
        return vocab

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize instruction text."""
        words = text.lower().split()
        tokens = [self.vocab.get('<start>')]
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab['<unk>']))
        tokens.append(self.vocab.get('<end>'))

        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return torch.tensor(tokens, dtype=torch.long)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        from PIL import Image
        import torchvision.transforms as T

        full_path = self.data_dir / image_path

        if full_path.exists():
            image = Image.open(full_path).convert('RGB')
        else:
            # Return dummy image if file not found
            image = Image.new('RGB', self.image_size, color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        else:
            transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)

        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, step_idx = self.samples[idx]
        episode = self.episodes[ep_idx]
        steps = episode['steps']

        # Get current and history frames
        images = []
        for i in range(self.history_length):
            hist_idx = max(0, step_idx - self.history_length + 1 + i)
            image_path = steps[hist_idx].get('image_path', '')
            images.append(self._load_image(image_path))

        images = torch.stack(images)  # [T, C, H, W]

        # Get instruction tokens
        instruction = episode.get('instruction', '')
        tokens = self._tokenize(instruction)

        # Get action
        action = steps[step_idx].get('action', [0.0] * 7)
        action = torch.tensor(action, dtype=torch.float32)

        return {
            'images': images,
            'tokens': tokens,
            'action': action,
            'episode_id': episode.get('episode_id', ''),
            'step_idx': step_idx
        }


class ActionLoss(nn.Module):
    """
    Loss function for VLA action prediction.

    Combines MSE loss with optional auxiliary losses.
    """

    def __init__(
        self,
        action_dim: int = 7,
        position_weight: float = 1.0,
        rotation_weight: float = 0.5,
        gripper_weight: float = 1.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.gripper_weight = gripper_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action loss.

        Args:
            pred: [B, action_dim] predicted actions
            target: [B, action_dim] target actions

        Returns:
            Dictionary with loss components
        """
        # Split into components
        pred_pos = pred[:, :3]
        pred_rot = pred[:, 3:6]
        pred_grip = pred[:, 6:]

        target_pos = target[:, :3]
        target_rot = target[:, 3:6]
        target_grip = target[:, 6:]

        # Compute losses
        pos_loss = F.mse_loss(pred_pos, target_pos)
        rot_loss = F.mse_loss(pred_rot, target_rot)
        grip_loss = F.binary_cross_entropy_with_logits(
            pred_grip,
            (target_grip > 0.5).float()
        ) if pred_grip.shape[-1] == 1 else F.mse_loss(pred_grip, target_grip)

        # Weighted total
        total_loss = (
            self.position_weight * pos_loss +
            self.rotation_weight * rot_loss +
            self.gripper_weight * grip_loss
        )

        return {
            'total': total_loss,
            'position': pos_loss,
            'rotation': rot_loss,
            'gripper': grip_loss
        }


class VLATrainer:
    """
    Trainer for VLA models.

    Handles training loop, evaluation, logging, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            model: VLA model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.criterion = ActionLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None

        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_val_loss = float('inf')

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train(self, epochs: Optional[int] = None, save_dir: Optional[str] = None):
        """
        Train the model.

        Args:
            epochs: Number of epochs (uses config if not specified)
            save_dir: Directory for saving checkpoints
        """
        epochs = epochs or self.config.epochs
        save_dir = Path(save_dir) if save_dir else Path('checkpoints')
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Training epoch
            train_loss = self._train_epoch(epoch)
            self.metrics['train_loss'].append(train_loss)

            # Validation
            if self.val_loader and epoch % self.config.eval_every == 0:
                val_loss = self._validate()
                self.metrics['val_loss'].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(save_dir / 'best_model.pt', epoch)
                    logger.info(f"New best model! Val loss: {val_loss:.4f}")

            # Regular checkpoint
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pt', epoch)

            # Update scheduler
            self.scheduler.step()

        # Save final model
        self._save_checkpoint(save_dir / 'final_model.pt', epochs)
        logger.info("Training complete!")

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            target = batch['action'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    pred = self.model(images, tokens)
                    losses = self.criterion(pred, target)
                    loss = losses['total']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(images, tokens)
                losses = self.criterion(pred, target)
                loss = losses['total']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item()

            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f} "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            target = batch['action'].to(self.device)

            pred = self.model(images, tokens)
            losses = self.criterion(pred, target)
            total_loss += losses['total'].item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")

        return avg_loss

    def _save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': dict(self.metrics)
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.metrics = defaultdict(list, checkpoint.get('metrics', {}))

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

        return checkpoint['epoch']


def create_demo_dataset(
    num_episodes: int = 10,
    steps_per_episode: int = 50,
    output_dir: str = 'demo_data'
) -> str:
    """
    Create synthetic demonstration data for testing.

    Args:
        num_episodes: Number of episodes to create
        steps_per_episode: Steps per episode
        output_dir: Output directory

    Returns:
        Path to created data directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    instructions = [
        "pick up the red cup",
        "place the block on the table",
        "grab the blue ball",
        "move to the shelf",
        "open the drawer"
    ]

    for ep_idx in range(num_episodes):
        episode = {
            'episode_id': f'{ep_idx:04d}',
            'instruction': instructions[ep_idx % len(instructions)],
            'steps': []
        }

        for step_idx in range(steps_per_episode):
            step = {
                'image_path': f'images/{ep_idx:04d}_{step_idx:04d}.jpg',
                'action': np.random.randn(7).tolist(),
                'state': np.random.randn(8).tolist()
            }
            episode['steps'].append(step)

        with open(output_path / f'episode_{ep_idx:04d}.json', 'w') as f:
            json.dump(episode, f, indent=2)

    logger.info(f"Created {num_episodes} demo episodes in {output_path}")
    return str(output_path)


if __name__ == '__main__':
    # Demo training setup
    print("VLA Training Demo")
    print("=" * 50)

    # Create demo data
    data_dir = create_demo_dataset(num_episodes=5, steps_per_episode=20)

    # Create dataset and loader
    dataset = RobotDemoDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create model
    from .simple_vla import create_vla_model
    model = create_vla_model('small', action_dim=7)

    # Create trainer
    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        device='cpu',
        mixed_precision=False,
        log_every=5
    )
    trainer = VLATrainer(model, train_loader, config=config)

    # Train
    print("\nStarting demo training...")
    trainer.train(save_dir='demo_checkpoints')
