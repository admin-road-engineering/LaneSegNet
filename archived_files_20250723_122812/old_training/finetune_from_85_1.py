#!/usr/bin/env python3
"""
TRUE Fine-Tuning Script - Load 85.1% mIoU checkpoint and continue
This script properly loads the Epoch 50 checkpoint and continues training from Epoch 51
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
import random
from datetime import datetime
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Import all the classes from premium_gpu_train.py
import sys
sys.path.append('scripts')
from premium_gpu_train import (
    EnhancedDiceFocalLoss, 
    PremiumLaneNet, 
    PremiumDataset,
    premium_gpu_training  # We'll modify this
)

def finetune_from_85_1_checkpoint():
    """
    Load the 85.1% mIoU checkpoint and continue fine-tuning
    """
    print("=" * 70)
    print("TRUE FINE-TUNING FROM 85.1% mIoU CHECKPOINT")
    print("Loading Epoch 50 checkpoint and continuing from Epoch 51")
    print("=" * 70)
    print()

    # Check if checkpoint exists
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        print("Cannot proceed with fine-tuning!")
        return None, None

    # GPU setup
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return None, None

    device = torch.device("cuda")
    scaler = GradScaler()
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()

    # Load checkpoint first to get the exact state
    print("Loading 85.1% mIoU checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} parameters")
        else:
            print(f"  {key}: {checkpoint[key]}")
    
    # Get checkpoint info
    checkpoint_epoch = checkpoint.get('epoch', 50)
    checkpoint_miou = checkpoint.get('best_miou', 0.85) * 100  # Convert to percentage
    
    print(f"SUCCESS: Loaded checkpoint from Epoch {checkpoint_epoch}")
    print(f"SUCCESS: Best mIoU in checkpoint: {checkpoint_miou:.1f}%")
    print()

    # Create model with exact same architecture
    class_weights = torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32)
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)
    
    # Load the checkpoint state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("SUCCESS: Successfully loaded model weights from checkpoint")
    except Exception as e:
        print(f"ERROR: Error loading model weights: {e}")
        return None, None

    # FINE-TUNING PARAMETERS (much lower learning rate)
    fine_tune_lr = 1e-5  # Very low learning rate for fine-tuning
    dice_weight = 0.694   # Use optimized dice weight from checkpoint
    
    print(f"FINE-TUNING CONFIGURATION:")
    print(f"  Learning Rate: {fine_tune_lr:.2e} (fine-tuning)")
    print(f"  Dice Weight: {dice_weight:.3f}")
    print(f"  Starting Epoch: {checkpoint_epoch + 1}")
    print(f"  Target Epochs: {checkpoint_epoch + 1} to {checkpoint_epoch + 30}")
    print(f"  Goal: Improve beyond {checkpoint_miou:.1f}% mIoU")
    print()

    # Enhanced loss function
    focal_weight = 1.0 - dice_weight
    criterion = EnhancedDiceFocalLoss(
        alpha=1.0, 
        gamma=2.0, 
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        class_weights=class_weights,
        label_smoothing=0.05
    ).to(device)

    # Fine-tuning optimizer with very low learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=fine_tune_lr,  # Very low LR for fine-tuning
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Update learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tune_lr
            print("SUCCESS: Loaded optimizer state and updated LR for fine-tuning")
        except:
            print("WARNING: Could not load optimizer state, starting fresh optimizer")

    # Gentle scheduler for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=30,  # 30 epoch fine-tuning
        eta_min=1e-7
    )

    # Load datasets
    train_dataset = PremiumDataset("data/ael_mmseg/img_dir/train", "data/ael_mmseg/ann_dir/train", mode='train')
    val_dataset = PremiumDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Smaller batch for fine-tuning stability
        shuffle=True, 
        num_workers=0,  # Fix Windows multiprocessing issue
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=0,  # Fix Windows multiprocessing issue
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: 8 (reduced for fine-tuning stability)")
    print()

    # Fine-tuning tracking
    best_miou = checkpoint_miou / 100.0  # Convert back to decimal
    best_balanced_score = checkpoint.get('best_balanced_score', 0.7)
    start_epoch = checkpoint_epoch + 1
    
    class_names = ['background', 'white_solid', 'white_dashed']
    
    # Fine-tuning loop
    print("STARTING FINE-TUNING...")
    print(f"Starting from mIoU: {best_miou*100:.1f}%")
    print()
    
    for epoch in range(start_epoch, start_epoch + 30):  # 30 epochs of fine-tuning
        print(f"Fine-tuning Epoch {epoch}/79:")
        
        # Training phase
        model.train()
        train_losses = []
        train_dice_losses = []
        train_focal_losses = []
        
        progress_bar = tqdm(train_loader, desc="Fine-tuning")
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
            train_dice_losses.append(dice_loss.item())
            train_focal_losses.append(focal_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}',
                'Focal': f'{focal_loss.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Fine-tuning Validation")
            for images, masks in progress_bar:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                
                val_losses.append(loss.item())
                
                # Collect predictions for metrics
                predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate per-class IoU
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_id in range(3):
            pred_class = (all_predictions == class_id)
            target_class = (all_targets == class_id)
            
            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()
            
            if union > 0:
                iou = intersection / union
                precision = intersection / pred_class.sum() if pred_class.sum() > 0 else 0
                recall = intersection / target_class.sum() if target_class.sum() > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                iou = precision = recall = f1 = 0
            
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Calculate overall metrics
        overall_miou = np.mean(ious)
        lane_classes_miou = np.mean(ious[1:])  # Exclude background
        lane_classes_f1 = np.mean(f1_scores[1:])
        balanced_score = (overall_miou + lane_classes_miou + lane_classes_f1) / 3
        
        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"  Train Loss: {np.mean(train_losses):.4f} (Hybrid: Dice+Focal+Lovász+Edge+Smooth)")
        print(f"    ├─ Dice: {np.mean(train_dice_losses):.4f}, Focal: {np.mean(train_focal_losses):.4f}")
        print(f"  Val Loss: {np.mean(val_losses):.4f}")
        print(f"  Overall mIoU: {overall_miou*100:.1f}%")
        print(f"  Lane Classes mIoU: {lane_classes_miou*100:.1f}%")
        print(f"  Lane Classes F1: {lane_classes_f1*100:.1f}%")
        print(f"  Balanced Score: {balanced_score*100:.1f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Per-class details
        print(f"  Detailed Per-class Metrics:")
        for i, class_name in enumerate(class_names):
            print(f"    {class_name}: IoU {ious[i]*100:.1f}%, Prec {precisions[i]*100:.1f}%, Rec {recalls[i]*100:.1f}%, F1 {f1_scores[i]*100:.1f}%")
        
        # Check for improvement (save if ANY improvement)
        improved = overall_miou > best_miou
        if improved:
            improvement = (overall_miou - best_miou) * 100
            best_miou = overall_miou
            best_balanced_score = balanced_score
            
            # Save improved model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"model_backups/option3_epoch{epoch}_finetune_{overall_miou*100:.1f}miou_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_balanced_score': best_balanced_score,
                'class_ious': ious,
                'learning_rate': current_lr,
                'improvement_from_85_1': improvement,
                'fine_tuning': True
            }
            
            torch.save(checkpoint_data, backup_dir / "option3_finetune_model.pth")
            
            # Save to separate working checkpoint (don't overwrite other approaches)
            torch.save(checkpoint_data, "work_dirs/option3_finetune_best_model.pth")
            
            print(f"  IMPROVEMENT: +{improvement:.2f}% -> {overall_miou*100:.1f}% mIoU")
            print(f"  SAVED to: {backup_dir}")
            
            # Save detailed results
            with open(backup_dir / f"OPTION3_FINETUNE_EPOCH{epoch}_RECORD.json", "w") as f:
                json.dump({
                    'epoch': epoch,
                    'miou': overall_miou,
                    'balanced_score': balanced_score,
                    'lane_miou': lane_classes_miou,
                    'lane_f1': lane_classes_f1,
                    'class_ious': ious,
                    'improvement_from_85_1': improvement,
                    'fine_tuning_lr': current_lr
                }, f, indent=2)
        else:
            print(f"  No improvement (best: {best_miou*100:.1f}%)")
        
        print("-" * 80)
        
        # Update scheduler
        scheduler.step()
    
    print("=" * 70)
    print("FINE-TUNING COMPLETE!")
    print(f"Final best mIoU: {best_miou*100:.1f}%")
    print(f"Improvement from 85.1%: {(best_miou*100 - 85.1):.2f}%")
    print("=" * 70)
    
    return best_miou, best_balanced_score

if __name__ == "__main__":
    finetune_from_85_1_checkpoint()