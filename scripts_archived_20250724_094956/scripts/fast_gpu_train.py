#!/usr/bin/env python3
"""
Fast GPU Training - Optimized for speed while maintaining quality
Based on the premium training but with speed optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# Import base components
import sys
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset, EnhancedDiceFocalLoss

def fast_gpu_training(learning_rate=0.001, dice_weight=0.6, approach="fast_option1"):
    """
    Fast GPU training with speed optimizations
    """
    print("=" * 70)
    print(f"FAST GPU TRAINING - {approach.upper()}")
    print("Speed-optimized while maintaining quality")
    print("=" * 70)
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

    # Load checkpoint if available
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    start_epoch = 0
    best_miou = 0
    best_balanced_score = 0
    
    if checkpoint_path.exists():
        print("Loading 85.1% mIoU checkpoint for continuation...")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint.get('epoch', 49) + 1
        best_miou = checkpoint.get('best_miou', 0.851)
        best_balanced_score = checkpoint.get('best_balanced_score', 0.77)
        print(f"SUCCESS: Loaded checkpoint from epoch {checkpoint.get('epoch', 49)}")
        print(f"SUCCESS: Starting from {best_miou*100:.1f}% mIoU baseline")
    else:
        print("No checkpoint found - starting fresh")
    print()

    # Fast model setup
    class_weights = torch.tensor([0.1, 5.0, 5.0], dtype=torch.float32)
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)
    
    # Load checkpoint weights if available
    if checkpoint_path.exists() and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("SUCCESS: Loaded model weights from checkpoint")

    # Fast loss function - optimized parameters for each approach
    if approach == "fast_option2":
        # Enhanced parameters for Option 2 (more aggressive)
        criterion = EnhancedDiceFocalLoss(
            alpha=1.0, 
            gamma=2.5,  # Higher gamma for harder examples
            dice_weight=dice_weight,
            focal_weight=1.0 - dice_weight,
            class_weights=class_weights,
            label_smoothing=0.1  # More smoothing
        ).to(device)
        print("Enhanced loss enabled: DiceFocal with aggressive parameters")
    else:
        # Standard parameters for Option 1 (proven settings)
        criterion = EnhancedDiceFocalLoss(
            alpha=1.0, 
            gamma=2.0, 
            dice_weight=dice_weight,
            focal_weight=1.0 - dice_weight,
            class_weights=class_weights,
            label_smoothing=0.05
        ).to(device)
        print("Standard loss enabled: DiceFocal with proven parameters")

    # Fast optimizer
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
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"SUCCESS: Loaded optimizer state, updated LR to {learning_rate:.2e}")
        except:
            print("WARNING: Could not load optimizer state")

    # Fast scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,
        T_mult=2,  # Must be integer >= 1
        eta_min=1e-7
    )

    # FAST DATA LOADING - Key optimization
    train_dataset = PremiumDataset("data/ael_mmseg/img_dir/train", "data/ael_mmseg/ann_dir/train", mode='train')
    val_dataset = PremiumDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    
    # Optimized data loaders for speed
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"FAST TRAINING SETUP:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: 16 (larger for speed)")
    print(f"  Workers: 4 train / 2 val")
    print(f"  Approach: {approach}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Target epochs: 20")
    print()

    class_names = ['background', 'white_solid', 'white_dashed']
    
    # FAST TRAINING LOOP
    print("STARTING FAST TRAINING...")
    print(f"Baseline: {best_miou*100:.1f}% mIoU")
    print()
    
    for epoch in range(20):
        actual_epoch = start_epoch + epoch
        print(f"Fast Epoch {actual_epoch+1} ({epoch+1}/20):")
        
        epoch_start_time = time.time()
        
        # Fast training phase
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc="Fast Training", leave=False)
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
        
        # Fast validation
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
        
        # Quick metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Fast IoU
        ious = []
        for class_id in range(3):
            pred_class = (all_predictions == class_id)
            target_class = (all_targets == class_id)
            
            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        overall_miou = np.mean(ious)
        lane_classes_miou = np.mean(ious[1:])
        balanced_score = overall_miou * 0.7 + lane_classes_miou * 0.3
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # Fast results
        print(f"  Train: {np.mean(train_losses):.4f}, Val: {np.mean(val_losses):.4f}")
        print(f"  mIoU: {overall_miou*100:.1f}%, Lanes: {lane_classes_miou*100:.1f}%, Time: {epoch_time:.1f}s")
        
        # Fast saving
        if overall_miou > best_miou:
            improvement = (overall_miou - best_miou) * 100
            best_miou = overall_miou
            best_balanced_score = balanced_score
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"model_backups/{approach}_epoch{actual_epoch+1}_{overall_miou*100:.1f}miou_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_balanced_score': best_balanced_score,
                'approach': approach
            }
            
            torch.save(checkpoint_data, backup_dir / f"{approach}_model.pth")
            torch.save(checkpoint_data, f"work_dirs/{approach}_best_model.pth")
            
            print(f"  IMPROVEMENT: +{improvement:.2f}% -> {overall_miou*100:.1f}% mIoU")
        else:
            print(f"  No improvement (best: {best_miou*100:.1f}%)")
        
        scheduler.step()
        print()
    
    print("=" * 70)
    print(f"FAST {approach.upper()} COMPLETE!")
    print(f"Final best mIoU: {best_miou*100:.1f}%")
    print("=" * 70)
    
    return best_miou, best_balanced_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', default='fast_option1')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dice', type=float, default=0.6)
    args = parser.parse_args()
    
    fast_gpu_training(args.lr, args.dice, args.approach)