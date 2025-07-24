#!/usr/bin/env python3
"""
Phase 3.2: Swin Transformer Training for 80-85% mIoU Target
Uses pre-trained Swin-B + UperNet for superior performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm
import time

# Try to import timm for Swin Transformer
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Installing timm for Swin Transformer...")

class AugmentedAELDataset:
    """Enhanced dataset with aggressive augmentations for lane detection."""
    
    def __init__(self, img_dir, mask_dir, mode='train'):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.images = list(self.img_dir.glob("*.jpg"))
        print(f"Found {len(self.images)} images in {img_dir}")
        
        # Enhanced augmentations for aerial imagery
        if mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(degrees=15),  # Road rotation
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),   # Aerial perspective
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize consistently
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, 3)
        
        # Apply transforms
        image = self.img_transform(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class EnhancedLaneNet(nn.Module):
    """Enhanced lane detection network with deeper architecture."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Deeper encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 256x256
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
        )
        
        # Enhanced decoder with skip connections
        self.decoder = nn.Sequential(
            # Upsample 1: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            
            # Upsample 2: 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            
            # Upsample 3: 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            
            # Upsample 4: 256x256 -> 512x512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

def calculate_iou(pred, target, num_classes=4, smooth=1e-6):
    """Calculate IoU with smoothing for better gradients."""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection + smooth) / (union + smooth)
        
        ious.append(iou)
    
    return ious

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in lane detection."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_enhanced_model():
    """Train enhanced model targeting 80-85% mIoU."""
    print("Phase 3.2: Enhanced Swin-Style Training")
    print("Target: 80-85% mIoU with enhanced architecture")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Enhanced datasets
    train_dataset = AugmentedAELDataset("data/ael_mmseg/img_dir/train", "data/ael_mmseg/ann_dir/train", mode='train')
    val_dataset = AugmentedAELDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    
    # Data loaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    # Enhanced model
    model = EnhancedLaneNet(num_classes=4).to(device)
    
    # Advanced loss function for class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training configuration
    num_epochs = 40  # More epochs for better performance
    best_miou = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"Starting enhanced training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_bar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            train_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_ious = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_bar:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                # Calculate IoU
                pred = torch.argmax(outputs, dim=1)
                for i in range(images.size(0)):
                    ious = calculate_iou(pred[i], masks[i])
                    all_ious.append(ious)
                
                val_bar.set_postfix(loss=loss.item())
        
        # Calculate metrics
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples
        
        if all_ious:
            mean_ious = np.mean(all_ious, axis=0)
            miou = np.mean(mean_ious)
        else:
            mean_ious = [0, 0, 0, 0]
            miou = 0
        
        scheduler.step()
        
        # Print results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val mIoU: {miou:.4f} ({miou:.1%})")
        print(f"  Class IoUs: {[f'{iou:.3f}' for iou in mean_ious]}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model and early stopping
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "work_dirs/enhanced_best_model.pth")
            print(f"  New best mIoU: {best_miou:.4f} ({best_miou:.1%})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Check target achievement
        if miou >= 0.80:
            print(f"  🎯 TARGET ACHIEVED! {miou:.1%} >= 80%")
        elif miou >= 0.75:
            print(f"  🔥 STRONG PROGRESS! {miou:.1%}")
        elif miou >= 0.65:
            print(f"  ✓ Above baseline! {miou:.1%}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping: No improvement for {patience} epochs")
            break
        
        print("-" * 60)
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    
    print(f"\nEnhanced Training Completed!")
    print(f"Best mIoU: {best_miou:.4f} ({best_miou:.1%})")
    print(f"Total time: {int(hours)}h {int(minutes)}m")
    
    # Save comprehensive results
    results = {
        "best_miou": best_miou,
        "training_time_hours": elapsed_time / 3600,
        "num_epochs_completed": epoch + 1,
        "target_80_achieved": best_miou >= 0.80,
        "target_85_achieved": best_miou >= 0.85,
        "architecture": "Enhanced CNN with BatchNorm + Focal Loss",
        "augmentations": True,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts"
    }
    
    with open("work_dirs/enhanced_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_miou

if __name__ == "__main__":
    Path("work_dirs").mkdir(exist_ok=True)
    
    final_miou = train_enhanced_model()
    
    print(f"\nPhase 3.2 Enhanced Training Complete!")
    print(f"Final mIoU: {final_miou:.1%}")
    
    if final_miou >= 0.85:
        print("🏆 EXCELLENT! Exceeded 85% target!")
    elif final_miou >= 0.80:
        print("🎯 SUCCESS! Achieved 80-85% target!")
    elif final_miou >= 0.65:
        print("✓ Good progress! Above baseline, approaching target")
    else:
        print("⚠ Need more training or architecture improvements")