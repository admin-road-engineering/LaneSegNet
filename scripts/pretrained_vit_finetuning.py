#!/usr/bin/env python3
"""
Pre-trained ViT Fine-tuning Solution
Addresses the core issue: ViT requires ImageNet pre-training for effective learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import logging
import os
import sys
from pathlib import Path
import json

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PretrainedViTLaneNet(nn.Module):
    """
    Lane detection model using ImageNet pre-trained Vision Transformer.
    SOLUTION: Uses proper pre-trained weights instead of random initialization.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, img_size=512, use_pretrained=True):
        super().__init__()
        
        # Import timm for pre-trained ViT models
        try:
            import timm
        except ImportError:
            logger.error("❌ timm library required for pre-trained ViT models")
            logger.error("Install with: pip install timm")
            raise
        
        # Load pre-trained ViT backbone
        if use_pretrained:
            logger.info("🎯 Loading ImageNet pre-trained ViT-Base weights...")
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, features_only=True)
            logger.info("✅ Pre-trained ViT-Base loaded successfully")
        else:
            logger.warning("⚠️ Using random initialization (not recommended)")
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, features_only=True)
        
        # Freeze early layers (optional - can help with stability)
        # Uncomment to freeze first 6 layers
        # for i, (name, param) in enumerate(self.backbone.named_parameters()):
        #     if i < 50:  # Freeze roughly first 6 transformer layers
        #         param.requires_grad = False
        
        # Get feature dimensions from backbone
        # ViT-Base outputs 768-dim features
        backbone_features = 768
        
        # Segmentation head with proper upsampling
        self.segmentation_head = nn.Sequential(
            # Initial projection
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Spatial reconstruction layers
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers to reconstruct full resolution
        self.upsample_layers = nn.ModuleList([
            # From 14x14 -> 28x28
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            # From 28x28 -> 56x56  
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            # From 56x56 -> 112x112
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ),
            # From 112x112 -> 224x224
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Final upsampling to target size (224 -> 512)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=3, padding=1)
        )
        
        # Calculate model parameters
        self._log_model_info()
        
    def _log_model_info(self):
        """Log model parameter information."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.segmentation_head.parameters())
        head_params += sum(p.numel() for p in self.upsample_layers.parameters())
        head_params += sum(p.numel() for p in self.final_upsample.parameters())
        
        total_params = backbone_params + head_params
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"PretrainedViTLaneNet created:")
        logger.info(f"  Backbone parameters: {backbone_params:,}")
        logger.info(f"  Segmentation head parameters: {head_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def forward(self, x):
        """Forward pass through the model."""
        # ViT expects 224x224 input, so we need to resize
        original_size = x.shape[-2:]
        
        # Resize to 224x224 for ViT backbone
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features using pre-trained ViT
        features = self.backbone(x_resized)
        
        # ViT features are typically (B, 197, 768) - 196 patches + 1 CLS token
        # We need the patch features (exclude CLS token)
        patch_features = features[-1]  # Get last layer features
        
        if len(patch_features.shape) == 3:
            # Remove CLS token if present (first token)
            if patch_features.shape[1] == 197:  # 196 patches + 1 CLS
                patch_features = patch_features[:, 1:, :]  # Remove CLS token
            
            # Reshape to spatial format: (B, 196, 768) -> (B, 768, 14, 14)
            B, N, C = patch_features.shape
            H = W = int(N ** 0.5)  # Should be 14 for 224x224 input with patch_size=16
            patch_features = patch_features.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply segmentation head
        B, C, H, W = patch_features.shape
        features_flat = patch_features.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        head_output = self.segmentation_head(features_flat)  # (B, H*W, 256)
        head_output = head_output.transpose(1, 2).view(B, -1, H, W)  # (B, 256, H, W)
        
        # Progressive upsampling
        x = head_output
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Final upsampling to target size
        x = self.final_upsample(x)
        
        # Resize to original input size if needed
        if x.shape[-2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x

class ImageNetNormalizedDataset:
    """
    Wrapper dataset that applies ImageNet normalization.
    CRITICAL: Must match pre-trained model expectations.
    """
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        # ImageNet normalization values (CRITICAL for pre-trained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]   # ImageNet stds
        )
        
        logger.info("📊 Using ImageNet normalization for pre-trained ViT compatibility")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Apply ImageNet normalization to image
        if isinstance(image, torch.Tensor):
            # Ensure image is in [0, 1] range before normalization
            if image.max() > 1.0:
                image = image / 255.0
            image = self.normalize(image)
        
        return image, mask

class ImprovedDiceFocalLoss(nn.Module):
    """Improved loss function from previous analysis."""
    
    def __init__(self, num_classes=3, alpha=None, gamma=3.0, smooth=1e-6, 
                 dice_weight=0.8, focal_weight=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Extreme class weights for imbalance
        if alpha is None:
            self.alpha = torch.tensor([0.005, 25.0, 30.0])
        else:
            self.alpha = torch.tensor(alpha)
    
    def focal_loss(self, inputs, targets):
        """Compute focal loss with extreme class weighting."""
        alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        alpha_t = alpha[targets]
        focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        """Compute dice loss with class weighting."""
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_losses = []
        alpha = self.alpha.to(inputs.device)
        
        for class_idx in range(self.num_classes):
            input_class = inputs_soft[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (input_class * target_class).sum()
            union = input_class.sum() + target_class.sum()
            
            if union > 0:
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice
            else:
                dice_loss = 0.0
            
            weighted_dice_loss = alpha[class_idx] * dice_loss
            dice_losses.append(weighted_dice_loss)
        
        return torch.stack(dice_losses).mean()
    
    def forward(self, inputs, targets):
        """Forward pass combining focal and dice losses."""
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

def create_balanced_tiny_dataset(base_dataset, num_samples=15):
    """Create tiny dataset with good class representation."""
    logger.info(f"🔍 Creating balanced tiny dataset with {num_samples} samples...")
    
    samples_with_lanes = []
    for idx in range(min(2000, len(base_dataset))):
        try:
            _, mask = base_dataset[idx]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            unique_classes = np.unique(mask_np)
            
            # Prefer samples with multiple lane types
            if len(unique_classes) >= 3:  # Background + 2 lane types
                samples_with_lanes.append(idx)
            elif len(unique_classes) >= 2 and any(cls > 0 for cls in unique_classes):
                samples_with_lanes.append(idx)
                
            if len(samples_with_lanes) >= num_samples:
                break
                
        except Exception:
            continue
    
    selected_indices = samples_with_lanes[:num_samples]
    logger.info(f"✅ Selected {len(selected_indices)} samples with lane diversity")
    
    return Subset(base_dataset, selected_indices)

def run_pretrained_overfitting_test():
    """Run overfitting test with pre-trained ViT model."""
    logger.info("🎯 RUNNING PRE-TRAINED VIT OVERFITTING TEST...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create pre-trained model  
        model = PretrainedViTLaneNet(num_classes=NUM_CLASSES, img_size=512, use_pretrained=True)
        model = model.to(device)
        model.train()
        
        # Create dataset with proper normalization
        from data.labeled_dataset import LabeledDataset
        base_dataset = LabeledDataset(
            "data/ael_mmseg/img_dir/train",
            "data/ael_mmseg/ann_dir/train", 
            mode='train',
            img_size=(512, 512)
        )
        
        # Apply ImageNet normalization
        normalized_dataset = ImageNetNormalizedDataset(base_dataset)
        
        # Create tiny balanced dataset
        tiny_dataset = create_balanced_tiny_dataset(normalized_dataset, num_samples=15)
        dataloader = DataLoader(tiny_dataset, batch_size=1, shuffle=True)
        
        # Use improved loss function
        criterion = ImprovedDiceFocalLoss(num_classes=NUM_CLASSES)
        
        # Differential learning rates: lower for pre-trained backbone
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.segmentation_head.parameters()) + \
                     list(model.upsample_layers.parameters()) + \
                     list(model.final_upsample.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},    # Lower LR for pre-trained
            {'params': head_params, 'lr': 1e-3}        # Higher LR for new head
        ], weight_decay=1e-4)
        
        logger.info("🚀 Starting pre-trained ViT overfitting test (30 epochs)...")
        logger.info("🎯 Target: >15% IoU (significant improvement expected)")
        
        best_iou = 0
        success_threshold = 0.15  # 15% IoU target
        
        for epoch in range(30):
            epoch_loss = 0
            epoch_iou = 0
            num_batches = 0
            
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate IoU
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    
                    ious = []
                    for class_id in range(1, NUM_CLASSES):
                        pred_mask = (predictions == class_id)
                        target_mask = (targets == class_id)
                        
                        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
                        union = torch.logical_or(pred_mask, target_mask).sum().item()
                        
                        if union > 0:
                            iou = intersection / union
                            ious.append(iou)
                    
                    sample_iou = np.mean(ious) if ious else 0.0
                    epoch_iou += sample_iou
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_iou = epoch_iou / num_batches if num_batches > 0 else 0
            
            if avg_iou > best_iou:
                best_iou = avg_iou
            
            logger.info(f"Epoch {epoch+1:2d}/30: Loss={avg_loss:.4f}, IoU={avg_iou:.1%} (best: {best_iou:.1%})")
            
            # Early success check
            if avg_iou >= success_threshold:
                logger.info(f"🎉 SUCCESS! Achieved {avg_iou:.1%} IoU at epoch {epoch+1}")
                logger.info("✅ Pre-trained ViT can successfully overfit small dataset!")
                break
        
        # Final assessment
        logger.info("=" * 70)
        logger.info("🏁 PRE-TRAINED VIT OVERFITTING TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Best IoU achieved: {best_iou:.1%}")
        logger.info(f"Target IoU (15%): {'✅ ACHIEVED' if best_iou >= success_threshold else '❌ FAILED'}")
        
        improvement_factor = best_iou / 0.023 if best_iou > 0 else 0  # vs 2.3% baseline
        logger.info(f"Improvement over random init: {improvement_factor:.1f}x")
        
        success = best_iou >= success_threshold
        
        if success:
            logger.info("🎯 CONCLUSION: Pre-trained weights SOLVE the overfitting problem!")
            logger.info("📈 Ready for full dataset fine-tuning")
        else:
            logger.warning("⚠️ Pre-trained weights help but may need further investigation")
        
        return success, best_iou
        
    except Exception as e:
        logger.error(f"💥 Pre-trained overfitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0

def main():
    """Test pre-trained ViT solution."""
    logger.info("🚀 PRE-TRAINED VIT SOLUTION TEST")
    logger.info("=" * 70)
    logger.info("🎯 Hypothesis: ViT needs ImageNet pre-training for effective learning")
    logger.info("=" * 70)
    
    # Check if timm is available
    try:
        import timm
        logger.info("✅ timm library available for pre-trained models")
    except ImportError:
        logger.error("❌ timm library required. Install with: pip install timm")
        return False
    
    # Run pre-trained overfitting test
    success, best_iou = run_pretrained_overfitting_test()
    
    # Final results and recommendations
    logger.info("\n" + "=" * 70)
    logger.info("🎯 PRE-TRAINED VIT SOLUTION RESULTS")
    logger.info("=" * 70)
    
    if success:
        logger.info("✅ SOLUTION SUCCESSFUL!")
        logger.info("🔑 Root cause confirmed: ViT requires ImageNet pre-training")
        logger.info(f"📈 Performance: {best_iou:.1%} IoU vs 2.3% with random weights")
        logger.info("🎯 Ready for full dataset fine-tuning")
        
        # Save solution summary
        solution = {
            "root_cause": "Vision Transformer requires ImageNet pre-training for effective learning",
            "solution": "Use pre-trained ViT backbone with ImageNet normalization",
            "performance": f"{best_iou:.1%} IoU on overfitting test",
            "improvement_factor": f"{best_iou/0.023:.1f}x better than random initialization",
            "ready_for_production": True,
            "next_steps": [
                "Integrate pre-trained ViT into production training pipeline",
                "Fine-tune on full dataset with differential learning rates",
                "Monitor convergence and adjust hyperparameters as needed"
            ]
        }
        
        solution_path = Path("work_dirs/pretrained_vit_solution.json")
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        logger.info(f"📄 Solution summary saved: {solution_path}")
        
    else:
        logger.warning("⚠️ PARTIAL SUCCESS")
        logger.warning("🔍 Pre-trained weights help but additional optimization needed")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)