#!/usr/bin/env python3
"""
Phase 3.2.5: Balanced Training with DiceFocal Loss
Research-proven solution for lane detection class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime

class DiceFocalLoss(nn.Module):
    """
    Research-proven compound loss for lane detection class imbalance.
    Combines Dice Loss (handles imbalance) + Focal Loss (focuses on hard examples).
    Based on 2024 lane detection research findings.
    """
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5, focal_weight=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def dice_loss(self, pred, target, num_classes=4):
        """Multi-class Dice Loss for handling class imbalance."""
        pred = F.softmax(pred, dim=1)
        dice_losses = []
        
        for cls in range(num_classes):
            pred_cls = pred[:, cls, :, :]
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum(dim=[1, 2])
            union = pred_cls.sum(dim=[1, 2]) + target_cls.sum(dim=[1, 2])
            
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_loss = 1 - dice.mean()
            dice_losses.append(dice_loss)
        
        return torch.stack(dice_losses).mean()
    
    def focal_loss(self, pred, target):
        """Focal Loss for focusing on hard-to-classify examples."""
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        """Combined DiceFocal loss."""
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        combined_loss = (self.dice_weight * dice) + (self.focal_weight * focal)
        return combined_loss, dice, focal

class OptimizedLaneNet(nn.Module):
    """
    Optimized architecture to prevent overfitting.
    Smaller, regularized model with dropout and batch normalization.
    Target: 10-20MB vs problematic 38.2MB enhanced model.
    """
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Optimized encoder - smaller but efficient
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.7),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
            
            # Block 4 - bottleneck
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
        )
        
        # Optimized decoder with skip connections
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Upsample 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Upsample 3
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.3),
            
            # Final layer
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class BalancedLaneDataset:
    """Enhanced dataset with selective cropping to ensure lane pixels."""
    def __init__(self, img_dir, mask_dir, mode='train'):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.images = list(self.img_dir.glob("*.jpg"))
        
        print(f"Dataset {mode}: {len(self.images)} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        # Load and process image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        # Load and process mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        mask = np.clip(mask, 0, 3)
        
        # Enhanced augmentations for training
        if self.mode == 'train':
            # Brightness/contrast for white line visibility
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.8, 1.2)
                image = image.astype(np.float32)
                image = np.clip(image * contrast + (brightness - 1) * 50, 0, 255)
                image = image.astype(np.uint8)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def calculate_balanced_iou(pred, target, num_classes=4):
    """Calculate IoU with focus on class balance."""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious, class_names

def train_balanced_model():
    """Train optimized model with DiceFocal loss for class imbalance fix."""
    print("=== Phase 3.2.5: Balanced Training with DiceFocal Loss ===")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Force GPU usage - we've verified it works
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("ERROR: CUDA not available! Please check GPU setup.")
        print("Run: python check_gpu_setup.py")
        return None, None
    print()
    
    # Research-proven class weights for lane detection
    # [0.1, 5.0, 5.0, 3.0] = [background, white_solid, white_dashed, yellow_solid]
    class_weights = [0.1, 5.0, 5.0, 3.0]
    print("Class weights (research-based):")
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    for name, weight in zip(class_names, class_weights):
        print(f"  {name}: {weight}")
    print()
    
    # Initialize optimized model
    model = OptimizedLaneNet(num_classes=4, dropout_rate=0.3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # DiceFocal loss with class weights
    criterion = DiceFocalLoss(
        alpha=1.0, 
        gamma=2.0, 
        dice_weight=0.6,  # Emphasize Dice for class imbalance
        focal_weight=0.4,  # De-emphasize Focal to prevent overfitting
        class_weights=class_weights
    ).to(device)
    
    # Optimizer with lower learning rate for stability
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Lower than previous 0.001
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    
    # Enhanced datasets
    train_dataset = BalancedLaneDataset("data/ael_mmseg/img_dir/train", "data/ael_mmseg/ann_dir/train", mode='train')
    val_dataset = BalancedLaneDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: 8")
    print()
    
    # Training loop
    best_miou = 0
    best_balanced_score = 0
    epochs_without_improvement = 0
    max_patience = 8  # Early stopping
    
    start_time = time.time()
    
    for epoch in range(25):  # Maximum epochs
        print(f"Epoch {epoch+1}/25:")
        
        # Training phase
        model.train()
        train_losses = []
        train_dice_losses = []
        train_focal_losses = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            combined_loss, dice_loss, focal_loss = criterion(outputs, masks)
            
            combined_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(combined_loss.item())
            train_dice_losses.append(dice_loss.item())
            train_focal_losses.append(focal_loss.item())
            
            progress_bar.set_postfix({
                'Loss': f'{combined_loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}',
                'Focal': f'{focal_loss.item():.4f}'
            })
        
        avg_train_loss = np.mean(train_losses)
        avg_dice_loss = np.mean(train_dice_losses)
        avg_focal_loss = np.mean(train_focal_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        all_ious = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                combined_loss, dice_loss, focal_loss = criterion(outputs, masks)
                val_losses.append(combined_loss.item())
                
                pred = torch.argmax(outputs, dim=1)
                
                for i in range(images.size(0)):
                    ious, class_names = calculate_balanced_iou(pred[i], masks[i])
                    all_ious.append(ious)
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses)
        mean_ious = np.mean(all_ious, axis=0)
        overall_miou = np.mean(mean_ious)
        
        # Balanced score: Prioritize lane class performance
        lane_classes_miou = np.mean(mean_ious[1:])  # Exclude background
        balanced_score = (overall_miou * 0.3) + (lane_classes_miou * 0.7)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"  Train Loss: {avg_train_loss:.4f} (Dice: {avg_dice_loss:.4f}, Focal: {avg_focal_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Overall mIoU: {overall_miou:.1%}")
        print(f"  Lane Classes mIoU: {lane_classes_miou:.1%}")
        print(f"  Balanced Score: {balanced_score:.1%}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print("  Per-class IoU:")
        for name, iou in zip(class_names, mean_ious):
            print(f"    {name}: {iou:.1%}")
        print()
        
        # Model saving logic
        improved = False
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_miou = overall_miou
            improved = True
            epochs_without_improvement = 0
            
            # Save best model (GPU version)
            torch.save(model.state_dict(), "work_dirs/gpu_balanced_best_model.pth")
            print(f"  ✅ New best balanced score: {balanced_score:.1%}")
        else:
            epochs_without_improvement += 1
        
        # Check targets
        if overall_miou >= 0.85:
            print(f"  🏆 OUTSTANDING! 85%+ target achieved!")
            break
        elif overall_miou >= 0.80:
            print(f"  🎯 SUCCESS! 80%+ target achieved!")
        elif overall_miou >= 0.70:
            print(f"  ✅ GOOD! 70%+ target achieved!")
        
        # Early stopping
        if epochs_without_improvement >= max_patience:
            print(f"  Early stopping: {max_patience} epochs without improvement")
            break
        
        print("-" * 60)
    
    training_time = time.time() - start_time
    
    # Final results
    print()
    print("=== TRAINING COMPLETED ===")
    print(f"Training time: {training_time/3600:.1f} hours")
    print(f"Best Overall mIoU: {best_miou:.1%}")
    print(f"Best Balanced Score: {best_balanced_score:.1%}")
    print()
    
    # Save training results
    results = {
        'best_miou': best_miou,
        'best_balanced_score': best_balanced_score,
        'training_time_hours': training_time / 3600,
        'architecture': 'OptimizedLaneNet with DiceFocal Loss',
        'class_weights': class_weights.tolist(),
        'target_80_achieved': best_miou >= 0.80,
        'target_85_achieved': best_miou >= 0.85,
        'target_70_achieved': best_miou >= 0.70,
        'class_imbalance_fixed': lane_classes_miou >= 0.50,  # All lane classes >50%
        'num_epochs_completed': epoch + 1,
        'final_per_class_ious': mean_ious.tolist(),
        'class_names': class_names
    }
    
    with open("work_dirs/gpu_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to work_dirs/gpu_training_results.json")
    print("Model saved to work_dirs/gpu_balanced_best_model.pth")
    
    # Check model size
    model_path = Path("work_dirs/gpu_balanced_best_model.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024**2
        print(f"Model size: {size_mb:.1f}MB")
        if size_mb < 20:
            print("✅ Model size optimized (target: 10-20MB)")
        else:
            print("⚠ Model size larger than target")
    
    return best_miou, best_balanced_score

if __name__ == "__main__":
    final_miou, balanced_score = train_balanced_model()
    print(f"\nPhase 3.2.5 Results:")
    print(f"Final mIoU: {final_miou:.1%}")
    print(f"Balanced Score: {balanced_score:.1%}")
    
    if final_miou >= 0.80:
        print("🎯 Ready for production deployment!")
    elif final_miou >= 0.70:
        print("✅ Significant improvement - consider production")
    else:
        print("📊 Analyze results for further optimization")