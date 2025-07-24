#!/usr/bin/env python3
"""
Full Dataset Premium Training
============================

Premium training on the complete 7,817 sample dataset with proper test holdout.
Target: Achieve 85%+ mIoU to surpass current 79.4% baseline.

Key improvements:
- Full 7,817 samples vs previous 6,799
- Proper 70/10/20 split with holdout test set
- Premium U-Net architecture with proven hyperparameters
- Enhanced loss function and class balancing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import json
import time
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, calculate_detailed_metrics

class FullDataset(Dataset):
    """Dataset class for full dataset training"""
    
    def __init__(self, img_dir, ann_dir, mode='train'):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.mode = mode
        
        # Get all image files
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
        
        # Verify matching annotation files exist
        self.valid_files = []
        for img_file in self.image_files:
            ann_file = self.ann_dir / f"{img_file.stem}.png"
            if ann_file.exists():
                self.valid_files.append(img_file.stem)
        
        print(f"Full Dataset {mode}: {len(self.valid_files)} valid samples")
        
        # Set up augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        sample_id = self.valid_files[idx]
        
        # Load image
        img_path = self.img_dir / f"{sample_id}.jpg"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.ann_dir / f"{sample_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        
        return transformed['image'], transformed['mask'].long()

def full_dataset_training():
    """Run premium training on full dataset"""
    
    print("FULL DATASET PREMIUM TRAINING")
    print("=" * 50)
    print("Target: 85%+ mIoU on complete 7,817 sample dataset")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FullDataset('data/full_ael_mmseg/img_dir/train', 'data/full_ael_mmseg/ann_dir/train', 'train')
    val_dataset = FullDataset('data/full_ael_mmseg/img_dir/val', 'data/full_ael_mmseg/ann_dir/val', 'val')
    test_dataset = FullDataset('data/full_ael_mmseg/img_dir/test', 'data/full_ael_mmseg/ann_dir/test', 'test')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")  
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model with proven architecture
    print(f"\nInitializing Premium U-Net model...")
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Enhanced loss function with class balancing
    class_weights = torch.tensor([0.1, 5.0, 5.0], device=device)  # Proven weights from previous training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with proven hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training configuration
    num_epochs = 60  # More epochs for full dataset
    best_miou = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {5e-4}")
    print(f"  Batch size: 16")
    print(f"  Class weights: {class_weights.tolist()}")
    
    print(f"\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += images.size(0)
            
            # Progress update every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU for batch
                predictions = torch.argmax(outputs, dim=1)
                for i in range(predictions.size(0)):
                    ious, _, _, _ = calculate_detailed_metrics(predictions[i].cpu(), masks[i].cpu())
                    all_ious.append(ious)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        mean_ious = np.mean(all_ious, axis=0)
        overall_miou = np.mean(mean_ious)
        lane_miou = np.mean(mean_ious[1:])  # Exclude background
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Overall mIoU: {overall_miou*100:.1f}%")
        print(f"  Lane mIoU: {lane_miou*100:.1f}%")
        print(f"  Background: {mean_ious[0]*100:.1f}%")
        print(f"  White Solid: {mean_ious[1]*100:.1f}%")
        print(f"  White Dashed: {mean_ious[2]*100:.1f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if overall_miou > best_miou:
            best_miou = overall_miou
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_lane_miou': lane_miou,
                'class_ious': mean_ious.tolist(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'training_config': {
                    'num_samples': len(train_dataset),
                    'batch_size': 16,
                    'learning_rate': 5e-4,
                    'class_weights': class_weights.tolist()
                }
            }
            
            torch.save(checkpoint, 'work_dirs/full_dataset_best_model.pth')
            print(f"  NEW BEST: {overall_miou*100:.1f}% mIoU (saved)")
            
            # Performance milestones
            if overall_miou >= 0.85:
                print(f"  TARGET ACHIEVED: {overall_miou*100:.1f}% >= 85% mIoU!")
            elif overall_miou >= 0.80:
                print(f"  EXCELLENT: {overall_miou*100:.1f}% >= 80% mIoU")
            elif overall_miou > 0.794:
                print(f"  IMPROVEMENT: {overall_miou*100:.1f}% > current 79.4%")
        else:
            patience_counter += 1
            print(f"  Best: {best_miou*100:.1f}% at epoch {best_epoch} (patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience: {patience})")
                break
        
        print("-" * 50)
    
    total_time = time.time() - start_time
    
    # Final results
    print(f"\nFULL DATASET TRAINING COMPLETE")
    print(f"=" * 50)
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Best mIoU: {best_miou*100:.1f}% at epoch {best_epoch}")
    print(f"Training samples used: {len(train_dataset)}")
    print(f"Model saved: work_dirs/full_dataset_best_model.pth")
    
    # Compare with baseline
    baseline_miou = 79.4
    improvement = (best_miou * 100) - baseline_miou
    
    if best_miou >= 0.85:
        print(f"SUCCESS: Target achieved! {best_miou*100:.1f}% >= 85%")
    elif improvement > 0:
        print(f"IMPROVEMENT: +{improvement:.1f}% over baseline ({baseline_miou}%)")
    else:
        print(f"Performance: {best_miou*100:.1f}% vs {baseline_miou}% baseline")
    
    return best_miou, best_epoch

def test_full_dataset_model():
    """Test the full dataset model on holdout test set"""
    
    print(f"\nTESTING FULL DATASET MODEL")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_dataset = FullDataset('data/full_ael_mmseg/img_dir/test', 'data/full_ael_mmseg/ann_dir/test', 'test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Load trained model
    model_path = 'work_dirs/full_dataset_best_model.pth'
    if not Path(model_path).exists():
        print("ERROR: Trained model not found!")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Testing trained model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation mIoU: {checkpoint.get('best_miou', 0)*100:.1f}%")
    print(f"Test samples: {len(test_dataset)}")
    
    # Run inference on test set
    all_ious = []
    test_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(predictions.size(0)):
                ious, _, _, _ = calculate_detailed_metrics(predictions[i].cpu(), masks[i].cpu())
                all_ious.append(ious)
            
            if batch_idx % 20 == 0:
                print(f"  Tested {batch_idx * images.size(0)}/{len(test_dataset)} samples")
    
    test_time = time.time() - test_time
    
    # Calculate final metrics
    mean_ious = np.mean(all_ious, axis=0)
    overall_miou = np.mean(mean_ious)
    lane_miou = np.mean(mean_ious[1:])
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"=" * 30)
    print(f"Test samples: {len(all_ious)}")
    print(f"Overall mIoU: {overall_miou*100:.1f}%")
    print(f"Lane mIoU: {lane_miou*100:.1f}%")
    print(f"Class performance:")
    print(f"  Background: {mean_ious[0]*100:.1f}%")
    print(f"  White Solid: {mean_ious[1]*100:.1f}%")
    print(f"  White Dashed: {mean_ious[2]*100:.1f}%")
    print(f"Test time: {test_time:.1f}s")
    
    # Compare with baseline
    baseline_miou = 79.4
    improvement = (overall_miou * 100) - baseline_miou
    
    if overall_miou >= 0.85:
        print(f"\nSUCCESS: Target achieved on test set!")
        print(f"  {overall_miou*100:.1f}% >= 85% target")
    elif improvement > 0:
        print(f"\nIMPROVEMENT on test set:")
        print(f"  +{improvement:.1f}% over {baseline_miou}% baseline")
    else:
        print(f"\nTest performance: {overall_miou*100:.1f}%")
    
    return overall_miou

def main():
    """Run full dataset training and testing"""
    
    # Phase 1: Train on full dataset
    best_miou, best_epoch = full_dataset_training()
    
    # Phase 2: Test on holdout set
    test_miou = test_full_dataset_model()
    
    print(f"\nFULL DATASET TRAINING SUMMARY")
    print("=" * 45)
    print(f"Training best: {best_miou*100:.1f}% mIoU")
    print(f"Test performance: {test_miou*100:.1f}% mIoU")
    print(f"Baseline comparison: 79.4% -> {test_miou*100:.1f}%")
    
    if test_miou >= 0.85:
        print(f"ACHIEVEMENT UNLOCKED: 85%+ mIoU target reached!")
    elif test_miou > 0.794:
        print(f"SUCCESS: Improved over baseline by {(test_miou*100 - 79.4):.1f}%")
    else:
        print(f"Result: Full dataset training completed")
    
    return test_miou

if __name__ == "__main__":
    result = main()