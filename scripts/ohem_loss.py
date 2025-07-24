#!/usr/bin/env python3
"""
Online Hard Example Mining (OHEM) Loss Implementation.
Focuses training on hardest pixels to improve difficult case performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss.
    Selects hardest examples for backpropagation.
    """
    
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=256, use_weight=True):
        super().__init__()
        self.thresh = thresh  # Loss threshold for hard examples
        self.min_kept = min_kept  # Minimum pixels to backprop
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        
        # Class weights for imbalanced datasets
        if use_weight:
            # Adjusted for 3-class lane detection: [background, white_solid, white_dashed]
            self.register_buffer('weight', torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32))
        else:
            self.weight = None
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - model predictions
            target: [B, H, W] - ground truth labels
        """
        B, C, H, W = pred.shape
        
        # Reshape for easier processing
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target_flat = target.reshape(-1)  # [B*H*W]
        
        # Remove ignored pixels
        valid_mask = target_flat != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Calculate per-pixel cross-entropy loss
        pixel_losses = F.cross_entropy(
            pred_valid, target_valid,
            weight=self.weight,
            reduction='none'
        )
        
        # Sort losses in descending order
        sorted_losses, _ = torch.sort(pixel_losses, descending=True)
        
        # Determine number of hard examples to keep
        n_valid = pixel_losses.numel()
        n_min_kept = min(self.min_kept, n_valid)
        
        # Dynamic threshold: keep examples above threshold OR minimum count
        if n_valid > n_min_kept and sorted_losses[n_min_kept - 1] > self.thresh:
            threshold = sorted_losses[n_min_kept - 1]
        elif n_valid > n_min_kept:
            threshold = self.thresh
        else:
            # Keep all if we have fewer than min_kept
            return pixel_losses.mean()
        
        # Create mask for hard examples
        hard_mask = pixel_losses >= threshold
        
        if hard_mask.sum() == 0:
            # Fallback: keep top min_kept examples
            _, top_indices = torch.topk(pixel_losses, n_min_kept)
            hard_mask = torch.zeros_like(pixel_losses, dtype=torch.bool)
            hard_mask[top_indices] = True
        
        # Return loss only on hard examples
        hard_losses = pixel_losses[hard_mask]
        return hard_losses.mean()

class OHEMDiceFocalLoss(nn.Module):
    """
    OHEM combined with DiceFocal loss for even better performance.
    Integrates OHEM with the successful DiceFocal loss from Phase 3.2.5.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6, 
                 ohem_thresh=0.7, ohem_min_kept=512, 
                 dice_weight=0.6, focal_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # OHEM parameters
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        
        # Class weights for imbalanced data
        self.register_buffer('class_weights', torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32))
    
    def dice_loss(self, pred, target, apply_softmax=True):
        """Compute Dice loss."""
        if apply_softmax:
            pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Compute Dice for each class
        dice_losses = []
        for c in range(pred.shape[1]):
            pred_c = pred[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_losses.append(1 - dice)
        
        # Weighted average by class weights
        dice_losses = torch.stack(dice_losses)
        class_weights_device = self.class_weights.to(dice_losses.device)
        weighted_dice = (dice_losses * class_weights_device).sum() / class_weights_device.sum()
        
        return weighted_dice
    
    def focal_loss_with_ohem(self, pred, target):
        """Compute Focal loss with OHEM selection."""
        B, C, H, W = pred.shape
        
        # Reshape for processing
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        target_flat = target.reshape(-1)  # [B*H*W]
        
        # Calculate per-pixel focal loss
        ce_loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting (ensure class_weights are on correct device)
        class_weights_device = self.class_weights.to(target_flat.device)
        alpha_weights = class_weights_device[target_flat]
        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss
        
        # Apply OHEM selection
        sorted_losses, _ = torch.sort(focal_loss, descending=True)
        n_total = focal_loss.numel()
        n_kept = min(self.ohem_min_kept, n_total)
        
        # Dynamic threshold
        if n_total > n_kept and sorted_losses[n_kept - 1] > self.ohem_thresh:
            threshold = sorted_losses[n_kept - 1]
        else:
            threshold = self.ohem_thresh
        
        # Select hard examples
        hard_mask = focal_loss >= threshold
        
        if hard_mask.sum() == 0:
            # Fallback to top examples
            _, top_indices = torch.topk(focal_loss, n_kept)
            hard_mask = torch.zeros_like(focal_loss, dtype=torch.bool)
            hard_mask[top_indices] = True
        
        return focal_loss[hard_mask].mean()
    
    def forward(self, pred, target):
        """Combined Dice + OHEM Focal loss."""
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss_with_ohem(pred, target)
        
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        return combined_loss

class AdaptiveOHEMLoss(nn.Module):
    """
    Adaptive OHEM that adjusts selection ratio based on training progress.
    Starts with more examples, gradually becomes more selective.
    """
    
    def __init__(self, base_loss='dice_focal', initial_ratio=0.5, final_ratio=0.1, 
                 total_epochs=100, warmup_epochs=10):
        super().__init__()
        self.base_loss = base_loss
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        # Initialize base loss
        if base_loss == 'ce':
            self.loss_fn = OHEMCrossEntropyLoss()
        elif base_loss == 'dice_focal':
            self.loss_fn = OHEMDiceFocalLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
        
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        """Update current epoch for adaptive behavior."""
        self.current_epoch = epoch
        
        # Calculate current selection ratio
        if epoch < self.warmup_epochs:
            # Warmup: use initial ratio
            current_ratio = self.initial_ratio
        else:
            # Linear decay from initial to final ratio
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            current_ratio = self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
        
        # Update OHEM parameters
        if hasattr(self.loss_fn, 'ohem_min_kept'):
            # Assume typical image has ~320k pixels
            total_pixels = 320000
            self.loss_fn.ohem_min_kept = int(total_pixels * current_ratio)
        
        return current_ratio
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)

def test_ohem_losses():
    """Test OHEM loss implementations."""
    print("Testing OHEM Loss Implementations")
    print("=" * 50)
    
    # Create dummy data
    B, C, H, W = 2, 3, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    # Test OHEM CrossEntropy
    ohem_ce = OHEMCrossEntropyLoss()
    loss_ce = ohem_ce(pred, target)
    print(f"OHEM CrossEntropy Loss: {loss_ce.item():.4f}")
    
    # Test OHEM DiceFocal
    ohem_df = OHEMDiceFocalLoss()
    loss_df = ohem_df(pred, target)
    print(f"OHEM DiceFocal Loss: {loss_df.item():.4f}")
    
    # Test Adaptive OHEM
    adaptive_ohem = AdaptiveOHEMLoss()
    for epoch in [0, 10, 50, 100]:
        ratio = adaptive_ohem.update_epoch(epoch)
        loss_adaptive = adaptive_ohem(pred, target)
        print(f"Epoch {epoch:3d} - Ratio: {ratio:.3f}, Loss: {loss_adaptive.item():.4f}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_ohem_losses()