#!/usr/bin/env python3
"""
Combined Dataset Training - Train on AEL + SS_Dense + SS_Multi_Lane
Enhanced training with multi-dataset approach for improved generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import cv2
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import base components
import sys
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, EnhancedDiceFocalLoss

class CombinedLaneDataset(Dataset):
    """Dataset class for combined lane detection datasets"""
    
    def __init__(self, images_dir, masks_dir, mode='train', target_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mode = mode
        self.target_size = target_size
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"CombinedLaneDataset {mode}: {len(self.image_files)} samples")
        
        # Setup augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        return image, mask.long()

def combined_dataset_training(learning_rate=0.001, dice_weight=0.6, epochs=30, approach="combined_training"):
    """
    Enhanced training on combined datasets
    """
    print("=" * 80)
    print(f"COMBINED DATASET TRAINING - {approach.upper()}")
    print("Training on AEL + SS_Dense + SS_Multi_Lane datasets")
    print("=" * 80)
    print()

    # GPU setup with optimizations
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return None, None

    device = torch.device("cuda")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Mixed precision for speed
    scaler = GradScaler()
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Optimizations: Enabled")
    print()

    # Load checkpoint if available (start from 85.1% model)
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    start_epoch = 0
    best_miou = 0
    best_balanced_score = 0
    
    if checkpoint_path.exists():
        print("Loading 85.1% mIoU checkpoint as starting point...")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint.get('epoch', 49) + 1
        best_miou = checkpoint.get('best_miou', 0.851)
        best_balanced_score = checkpoint.get('best_balanced_score', 0.77)
        print(f"SUCCESS: Starting from epoch {checkpoint.get('epoch', 49)}")
        print(f"SUCCESS: Starting from {best_miou*100:.1f}% mIoU baseline")
    else:
        print("No checkpoint found - starting fresh")
    print()

    # Enhanced model setup
    class_weights = torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32)
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)
    
    # Load checkpoint weights if available
    if checkpoint_path.exists() and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("SUCCESS: Loaded model weights from checkpoint")

    # Enhanced loss function for multi-dataset training
    criterion = EnhancedDiceFocalLoss(
        alpha=1.0, 
        gamma=2.0, 
        dice_weight=dice_weight,
        focal_weight=1.0 - dice_weight,
        class_weights=class_weights,
        label_smoothing=0.05
    ).to(device)

    # Optimizer with fine-tuning learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Load optimizer state if available
    if checkpoint_path.exists() and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Update learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"SUCCESS: Loaded optimizer state, updated LR to {learning_rate:.2e}")
        except:
            print("WARNING: Could not load optimizer state")

    # Scheduler for gradual learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=1e-7
    )

    # COMBINED DATASET LOADING
    combined_base = Path("data/combined_lane_dataset")
    
    train_dataset = CombinedLaneDataset(
        combined_base / "train" / "images", 
        combined_base / "train" / "masks", 
        mode='train'
    )
    val_dataset = CombinedLaneDataset(
        combined_base / "val" / "images", 
        combined_base / "val" / "masks", 
        mode='val'
    )
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=12,  # Slightly smaller for stability with diverse data
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=12,
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"COMBINED DATASET TRAINING SETUP:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: 12")
    print(f"  Workers: 4 train / 2 val")
    print(f"  Approach: {approach}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Target epochs: {epochs}")
    print()

    class_names = ['background', 'white_solid', 'white_dashed']
    
    # ENHANCED TRAINING LOOP
    print("STARTING COMBINED DATASET TRAINING...")
    print(f"Baseline: {best_miou*100:.1f}% mIoU")
    print(f"Target: Improve generalization across multiple datasets")
    print()
    
    training_history = []
    
    for epoch in range(epochs):
        actual_epoch = start_epoch + epoch
        print(f"Combined Epoch {actual_epoch+1} ({epoch+1}/{epochs}):")
        
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc="Multi-Dataset Training", leave=False)
        for images, masks in progress_bar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss, dice_loss, focal_loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                
                val_losses.append(loss.item())
                
                predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Per-class IoU
        ious = []
        for class_id in range(3):
            pred_class = (all_predictions == class_id)
            target_class = (all_targets == class_id)
            
            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        overall_miou = np.mean(ious)
        lane_classes_miou = np.mean(ious[1:])  # Only lane classes
        balanced_score = overall_miou * 0.7 + lane_classes_miou * 0.3
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # Detailed results
        print(f"  Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
        print(f"  mIoU: {overall_miou*100:.1f}%, Lane mIoU: {lane_classes_miou*100:.1f}%")
        print(f"  Class IoUs: BG={ious[0]*100:.1f}%, Solid={ious[1]*100:.1f}%, Dashed={ious[2]*100:.1f}%")
        print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Record training history
        epoch_data = {
            'epoch': actual_epoch + 1,
            'train_loss': float(np.mean(train_losses)),
            'val_loss': float(np.mean(val_losses)),
            'overall_miou': float(overall_miou),
            'lane_miou': float(lane_classes_miou),
            'class_ious': [float(iou) for iou in ious],
            'learning_rate': float(current_lr),
            'epoch_time': float(epoch_time)
        }
        training_history.append(epoch_data)
        
        # Save model if improved
        if overall_miou > best_miou:
            improvement = (overall_miou - best_miou) * 100
            best_miou = overall_miou
            best_balanced_score = balanced_score
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"model_backups/combined_{approach}_epoch{actual_epoch+1}_{overall_miou*100:.1f}miou_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_balanced_score': best_balanced_score,
                'approach': approach,
                'training_history': training_history,
                'dataset_info': 'AEL + SS_Dense + SS_Multi_Lane combined'
            }
            
            # Save model files
            torch.save(checkpoint_data, backup_dir / f"combined_{approach}_model.pth")
            torch.save(checkpoint_data, f"work_dirs/combined_{approach}_best_model.pth")
            
            # Save training history
            with open(backup_dir / "training_history.json", 'w') as f:
                json.dump(training_history, f, indent=2)
            
            print(f"  IMPROVEMENT: +{improvement:.2f}% -> {overall_miou*100:.1f}% mIoU")
        else:
            print(f"  No improvement (best: {best_miou*100:.1f}%)")
        
        scheduler.step()
        print()
    
    print("=" * 80)
    print(f"COMBINED {approach.upper()} TRAINING COMPLETE!")
    print(f"Final best mIoU: {best_miou*100:.1f}%")
    print(f"Trained on {len(train_dataset)} multi-dataset samples")
    print("=" * 80)
    
    return best_miou, best_balanced_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', default='combined_training')
    parser.add_argument('--lr', type=float, default=0.0005)  # Lower LR for fine-tuning
    parser.add_argument('--dice', type=float, default=0.6)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    combined_dataset_training(args.lr, args.dice, args.epochs, args.approach)