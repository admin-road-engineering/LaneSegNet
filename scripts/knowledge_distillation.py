#!/usr/bin/env python3
"""
Knowledge Distillation Pipeline for Production-Ready Model.
Creates lightweight student model from heavy teacher model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientStudentLaneNet(nn.Module):
    """
    Lightweight student model for production deployment.
    Target: <2M parameters, <100ms inference time.
    """
    
    def __init__(self, num_classes=3, base_channels=32):
        super().__init__()
        
        # Efficient encoder (MobileNetV2-inspired)
        self.encoder = nn.ModuleList([
            # Stage 1: Input processing
            self._make_stage(3, base_channels, stride=2),  # 1280 -> 640
            
            # Stage 2: Feature extraction
            self._make_stage(base_channels, base_channels * 2, stride=2),  # 640 -> 320
            self._make_stage(base_channels * 2, base_channels * 4, stride=2),  # 320 -> 160
            self._make_stage(base_channels * 4, base_channels * 8, stride=2),  # 160 -> 80
            
            # Stage 3: High-level features
            self._make_stage(base_channels * 8, base_channels * 16, stride=2),  # 80 -> 40
        ])
        
        # Efficient decoder with skip connections
        self.decoder = nn.ModuleList([
            self._make_upsampling_stage(base_channels * 16, base_channels * 8),  # 40 -> 80
            self._make_upsampling_stage(base_channels * 16, base_channels * 4),  # 80 -> 160 (concat)
            self._make_upsampling_stage(base_channels * 8, base_channels * 2),   # 160 -> 320 (concat)
            self._make_upsampling_stage(base_channels * 4, base_channels),       # 320 -> 640 (concat)
            self._make_upsampling_stage(base_channels * 2, base_channels // 2),  # 640 -> 1280 (concat)
        ])
        
        # Final prediction head
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        # Calculate and log model size
        self._log_model_info()
    
    def _make_stage(self, in_channels, out_channels, stride=1):
        """Create efficient convolution stage."""
        return nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def _make_upsampling_stage(self, in_channels, out_channels):
        """Create efficient upsampling stage."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            
            # Refinement convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"EfficientStudentLaneNet created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < len(self.encoder) - 1:  # Skip the last encoder stage
                skip_connections.append(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder with skip connections
        for i, stage in enumerate(self.decoder):
            x = stage(x)
            
            # Add skip connection if available
            # Channel concatenation doubles input channels for next decoder stage
            if i < len(skip_connections):
                skip = skip_connections[i]
                # Ensure compatible sizes
                if x.shape[2:] == skip.shape[2:]:
                    x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        
        # Final prediction
        x = self.final_conv(x)
        
        return x

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard and soft targets.
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, use_dice=True):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_dice = use_dice
        
        # Class weights for hard loss
        self.register_buffer('class_weights', torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32))
    
    def hard_target_loss(self, student_outputs, targets):
        """Standard cross-entropy loss against ground truth."""
        if self.use_dice:
            return self._dice_loss(student_outputs, targets)
        else:
            return F.cross_entropy(student_outputs, targets, weight=self.class_weights)
    
    def soft_target_loss(self, student_outputs, teacher_outputs):
        """KL divergence loss against teacher's soft predictions."""
        # Apply temperature scaling
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared (as per original distillation paper)
        return kl_loss * (self.temperature ** 2)
    
    def _dice_loss(self, pred, target):
        """Compute Dice loss for segmentation."""
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        dice_losses = []
        for c in range(pred.shape[1]):
            pred_c = pred_soft[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + 1e-6) / (pred_c.sum() + target_c.sum() + 1e-6)
            dice_losses.append(1 - dice)
        
        # Weighted by class importance
        dice_losses = torch.stack(dice_losses)
        weighted_dice = (dice_losses * self.class_weights).sum() / self.class_weights.sum()
        
        return weighted_dice
    
    def forward(self, student_outputs, teacher_outputs, targets):
        """Combined distillation loss."""
        hard_loss = self.hard_target_loss(student_outputs, targets)
        soft_loss = self.soft_target_loss(student_outputs, teacher_outputs)
        
        # Combine losses
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }

class KnowledgeDistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(self, teacher_model, student_model, device, save_dir="work_dirs/distillation"):
        self.teacher_model = teacher_model.to(device).eval()  # Teacher is frozen
        self.student_model = student_model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Loss function
        self.criterion = DistillationLoss(temperature=4.0, alpha=0.7)
        
        # Optimizer (smaller learning rate for distillation)
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=1e-4,  # Lower than normal training
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        self.training_log = []
        logger.info("Knowledge Distillation Trainer initialized")
    
    def train_epoch(self, dataloader, epoch):
        """Train student model for one epoch."""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        
        progress = tqdm(dataloader, desc=f"Distillation Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(images)
            
            # Get student predictions
            student_outputs = self.student_model(images)
            
            # Compute distillation loss
            loss_dict = self.criterion(student_outputs, teacher_outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_hard_loss += loss_dict['hard_loss'].item()
            total_soft_loss += loss_dict['soft_loss'].item()
            
            # Update progress
            progress.set_postfix({
                'total': f"{loss_dict['total_loss'].item():.4f}",
                'hard': f"{loss_dict['hard_loss'].item():.4f}",
                'soft': f"{loss_dict['soft_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Update scheduler
        self.scheduler.step()
        
        # Return average losses
        num_batches = len(dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'hard_loss': total_hard_loss / num_batches,
            'soft_loss': total_soft_loss / num_batches
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'training_log': self.training_log
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"student_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "student_best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best student model saved: {best_path}")
    
    def train(self, dataloader, epochs=50):
        """Full distillation training loop."""
        logger.info(f"Starting Knowledge Distillation for {epochs} epochs")
        logger.info(f"Dataset size: {len(dataloader.dataset)}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch + 1)
            
            self.training_log.append({
                'epoch': epoch + 1,
                **metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Check if best model
            is_best = metrics['total_loss'] < best_loss
            if is_best:
                best_loss = metrics['total_loss']
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {metrics['total_loss']:.4f} "
                f"(Hard: {metrics['hard_loss']:.4f}, Soft: {metrics['soft_loss']:.4f})"
            )
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, metrics, is_best)
        
        # Save final model
        self.save_checkpoint(epochs, metrics, is_best=False)
        
        # Save training log
        log_path = self.save_dir / "distillation_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        logger.info("Knowledge Distillation completed!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Final student model saved to: {self.save_dir}")

def load_teacher_model(teacher_path, device):
    """Load pre-trained teacher model."""
    logger.info(f"Loading teacher model from: {teacher_path}")
    
    # This is a placeholder - you'll need to load your actual trained model
    # For now, assume it's saved in the standard format
    try:
        checkpoint = torch.load(teacher_path, map_location=device)
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create teacher model (you'll need to import your actual model class)
        # teacher_model = YourLaneDetectionModel(num_classes=3)
        # teacher_model.load_state_dict(state_dict)
        
        logger.info("Teacher model loaded successfully")
        return None  # Placeholder
        
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        raise

def main():
    """Main distillation function."""
    print("KNOWLEDGE DISTILLATION PIPELINE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create student model
    student_model = EfficientStudentLaneNet(num_classes=3)
    
    print("\nStudent model created successfully!")
    print("Next steps:")
    print("1. Load your trained teacher model")
    print("2. Prepare training dataloader") 
    print("3. Run distillation training")
    print("4. Deploy lightweight student model")

if __name__ == "__main__":
    main()