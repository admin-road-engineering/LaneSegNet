#!/usr/bin/env python3
"""
Improved Loss Function for Extreme Class Imbalance
Addresses the critical 400:1 class imbalance issue identified in Phase 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImprovedDiceFocalLoss(nn.Module):
    """
    Improved loss function specifically designed for extreme class imbalance.
    Combines adaptive focal loss with dice loss and proper class weighting.
    """
    
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, smooth=1e-6, 
                 dice_weight=0.7, focal_weight=0.3, adaptive_weights=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.adaptive_weights = adaptive_weights
        
        # Aggressive class weights for extreme imbalance
        if alpha is None:
            # Weight background much less, lane classes much more
            self.alpha = torch.tensor([0.05, 10.0, 15.0])  # Background:Lane1:Lane2
        else:
            self.alpha = torch.tensor(alpha)
            
        logger.info(f"ImprovedDiceFocalLoss initialized with weights: {self.alpha.tolist()}")
    
    def focal_loss(self, inputs, targets):
        """Compute focal loss with adaptive class weighting."""
        # Ensure alpha is on the same device
        alpha = self.alpha.to(inputs.device)
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities and apply focal term
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights
        alpha_t = alpha[targets]
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        """Compute multi-class dice loss with class weighting."""
        # Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_losses = []
        alpha = self.alpha.to(inputs.device)
        
        for class_idx in range(self.num_classes):
            input_class = inputs[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (input_class * target_class).sum()
            union = input_class.sum() + target_class.sum()
            
            if union == 0:
                # No ground truth for this class
                dice_loss = 0.0
            else:
                dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            
            # Apply class weight
            weighted_dice_loss = alpha[class_idx] * dice_loss
            dice_losses.append(weighted_dice_loss)
        
        return torch.stack(dice_losses).mean()
    
    def forward(self, inputs, targets):
        """Forward pass combining focal and dice losses."""
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss

class BalancedSamplingLoss(nn.Module):
    """
    Loss function with balanced sampling to address extreme imbalance.
    """
    
    def __init__(self, num_classes=3, base_loss='focal', sample_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.sample_ratio = sample_ratio
        
        if base_loss == 'focal':
            # Aggressive weights for lanes
            self.criterion = ImprovedDiceFocalLoss(
                num_classes=num_classes,
                alpha=[0.02, 12.0, 18.0],  # Even more aggressive
                gamma=3.0,  # Higher gamma for harder focusing
                dice_weight=0.8,
                focal_weight=0.2
            )
        else:
            # Weighted cross-entropy fallback
            weights = torch.tensor([0.02, 12.0, 18.0])
            self.criterion = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, inputs, targets):
        """Forward pass with balanced sampling."""
        batch_size, num_classes, height, width = inputs.shape
        
        # Find pixels of each class
        background_mask = (targets == 0)
        lane_masks = [(targets == i) for i in range(1, self.num_classes)]
        
        # Sample balanced pixels
        sampled_indices = []
        
        # Find lane pixels (minority classes)
        lane_pixels = torch.zeros_like(targets, dtype=torch.bool)
        for lane_mask in lane_masks:
            lane_pixels |= lane_mask
        
        lane_indices = torch.nonzero(lane_pixels, as_tuple=False)
        
        if len(lane_indices) > 0:
            # Sample background pixels proportionally
            bg_indices = torch.nonzero(background_mask, as_tuple=False)
            
            # Sample ratio of background pixels relative to lane pixels
            num_lane_pixels = len(lane_indices)
            num_bg_to_sample = min(int(num_lane_pixels / self.sample_ratio), len(bg_indices))
            
            if num_bg_to_sample > 0:
                # Random sample background pixels
                bg_sample_idx = torch.randperm(len(bg_indices))[:num_bg_to_sample]
                sampled_bg_indices = bg_indices[bg_sample_idx]
                
                # Combine lane and sampled background pixels
                sampled_indices = torch.cat([lane_indices, sampled_bg_indices], dim=0)
            else:
                sampled_indices = lane_indices
            
            # Create balanced targets and inputs
            if len(sampled_indices) > 0:
                # Extract sampled pixels
                batch_idx = sampled_indices[:, 0]
                h_idx = sampled_indices[:, 1]
                w_idx = sampled_indices[:, 2]
                
                sampled_targets = targets[batch_idx, h_idx, w_idx]
                sampled_inputs = inputs[batch_idx, :, h_idx, w_idx].transpose(0, 1)
                
                # Compute loss on balanced sample
                return self.criterion(sampled_inputs, sampled_targets)
        
        # Fallback to full loss if sampling fails
        return self.criterion(inputs, targets)

def test_improved_loss():
    """Test the improved loss function on synthetic imbalanced data."""
    logger.info("🧪 Testing improved loss function...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create severely imbalanced synthetic data
    batch_size, img_size = 2, 256
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    targets = torch.zeros(batch_size, img_size, img_size, dtype=torch.long).to(device)
    
    # Create extreme imbalance: ~99% background, <1% lanes
    targets[0, 100:110, 100:150] = 1  # Small lane
    targets[1, 120:125, 120:140] = 2  # Very small lane
    
    # Calculate actual class distribution
    unique, counts = torch.unique(targets, return_counts=True)
    total_pixels = targets.numel()
    
    logger.info("Class distribution in test data:")
    for cls, count in zip(unique.tolist(), counts.tolist()):
        percentage = count / total_pixels * 100
        logger.info(f"  Class {cls}: {count} pixels ({percentage:.2f}%)")
    
    # Test different loss functions
    loss_functions = [
        ('Standard CrossEntropy', nn.CrossEntropyLoss()),
        ('Weighted CrossEntropy', nn.CrossEntropyLoss(weight=torch.tensor([0.02, 12.0, 18.0]).to(device))),
        ('Improved DiceFocal', ImprovedDiceFocalLoss(num_classes=3).to(device)),
        ('Balanced Sampling', BalancedSamplingLoss(num_classes=3, sample_ratio=0.05).to(device))
    ]
    
    # Simple test model
    test_model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, 1)
    ).to(device)
    
    logger.info("\nTesting loss functions:")
    
    for loss_name, loss_fn in loss_functions:
        logger.info(f"\n--- {loss_name} ---")
        
        try:
            test_model.eval()
            with torch.no_grad():
                outputs = test_model(images)
            
            # Calculate loss
            loss_value = loss_fn(outputs, targets)
            logger.info(f"Loss: {loss_value.item():.4f}")
            
            # Analyze predictions
            predictions = torch.argmax(outputs, dim=1)
            pred_unique, pred_counts = torch.unique(predictions, return_counts=True)
            
            logger.info(f"Predicted classes: {pred_unique.tolist()}")
            
            # Calculate IoU for each class
            for class_id in range(1, 3):
                pred_mask = (predictions == class_id)
                target_mask = (targets == class_id)
                
                intersection = (pred_mask & target_mask).sum().item()
                union = (pred_mask | target_mask).sum().item()
                
                if union > 0:
                    iou = intersection / union
                    logger.info(f"Class {class_id} IoU: {iou:.3f}")
                else:
                    logger.info(f"Class {class_id}: No ground truth")
        
        except Exception as e:
            logger.error(f"Error testing {loss_name}: {e}")
    
    logger.info("\n✅ Improved loss function test completed")

def create_production_loss_function(num_classes=3, loss_type='improved_dice_focal'):
    """
    Create the recommended production loss function based on analysis.
    """
    if loss_type == 'improved_dice_focal':
        return ImprovedDiceFocalLoss(
            num_classes=num_classes,
            alpha=[0.02, 15.0, 20.0],  # Aggressive lane weighting
            gamma=3.0,  # Strong focusing on hard examples
            dice_weight=0.8,  # Emphasize dice for segmentation
            focal_weight=0.2,
            smooth=1e-6
        )
    elif loss_type == 'balanced_sampling':
        return BalancedSamplingLoss(
            num_classes=num_classes,
            base_loss='focal',
            sample_ratio=0.03  # Very aggressive sampling
        )
    else:
        # Fallback to heavily weighted cross-entropy
        weights = torch.tensor([0.01, 20.0, 25.0])
        return nn.CrossEntropyLoss(weight=weights)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_improved_loss()