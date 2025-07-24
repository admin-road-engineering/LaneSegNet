#!/usr/bin/env python3
"""
Fine-Tuning Premium Model - Option 3
Continue from 85.1% mIoU Epoch 50 checkpoint
Target: 85.5-86.0% mIoU with conservative improvements
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
import shutil

# Enhanced imports for fine-tuning
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Enable fine-tuning optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Allow some randomness for better exploration

class EnhancedDiceFocalLoss(nn.Module):
    """
    Fine-tuning optimized loss: Gentler regularization
    """
    def __init__(self, alpha=1, gamma=1.5, dice_weight=0.65, focal_weight=0.35, 
                 class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma  # Reduced for fine-tuning
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing  # Reduced for fine-tuning
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        # Gentler edge detection for fine-tuning
        sobel_kernel = torch.tensor([[-0.5, -0.5, -0.5], [-0.5, 4, -0.5], [-0.5, -0.5, -0.5]], 
                                   dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_kernel', sobel_kernel)
    
    def dice_loss(self, pred, target, num_classes=3, smooth=1e-6):
        pred = F.softmax(pred, dim=1)
        dice_losses = []
        
        for class_idx in range(num_classes):
            pred_class = pred[:, class_idx, :, :]
            target_class = (target == class_idx).float()
            
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_loss = 1 - dice
            dice_losses.append(dice_loss.mean())
        
        return sum(dice_losses) / len(dice_losses)
    
    def focal_loss(self, pred, target, num_classes=3):
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, 
                                 label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Gentler edge loss for fine-tuning
        pred_edges = F.conv2d(F.softmax(pred, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True), 
                             self.sobel_kernel, padding=1)
        target_edges = F.conv2d((target > 0).float().unsqueeze(1), 
                               self.sobel_kernel, padding=1)
        edge_loss = F.mse_loss(pred_edges, target_edges) * 0.05  # Reduced for fine-tuning
        
        # Gentler smoothness loss
        diff_h = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        diff_w = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        smooth_loss = (diff_h.mean() + diff_w.mean()) * 0.02  # Reduced for fine-tuning
        
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal + 
                     edge_loss + 
                     smooth_loss)
        
        return total_loss, dice.item(), focal.item()

class FineTuningPremiumUNet(nn.Module):
    """Premium U-Net with fine-tuning optimizations"""
    def __init__(self, num_classes=3, dropout_rate=0.2):  # Reduced dropout for fine-tuning
        super().__init__()
        
        # Encoder with attention
        self.enc1 = self._make_layer(3, 64, dropout_rate)
        self.enc2 = self._make_layer(64, 128, dropout_rate)
        self.enc3 = self._make_layer(128, 256, dropout_rate)
        self.enc4 = self._make_layer(256, 512, dropout_rate)
        
        # Bottleneck with enhanced attention
        self.bottleneck = self._make_layer(512, 1024, dropout_rate, attention=True)
        
        # Decoder with skip connections
        self.dec4 = self._make_decoder_layer(1024, 512, dropout_rate)
        self.dec3 = self._make_decoder_layer(512, 256, dropout_rate)
        self.dec2 = self._make_decoder_layer(256, 128, dropout_rate)
        self.dec1 = self._make_decoder_layer(128, 64, dropout_rate)
        
        # Final classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _make_layer(self, in_channels, out_channels, dropout_rate, attention=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if attention:
            layers.append(self._attention_block(out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _attention_block(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final_conv(d1)

class FineTuningAELDataset(torch.utils.data.Dataset):
    """AEL Dataset with gentle augmentations for fine-tuning"""
    def __init__(self, data_file, is_training=True):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.is_training = is_training
        
        # Gentler augmentations for fine-tuning
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.3),  # Reduced
                transforms.RandomRotation(5, fill=0),     # Reduced
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Reduced
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = f"data/imgs/{item['id']}.jpg"
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = f"data/mask/{item['id']}.jpg"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Resize to training size
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to class indices (0=background, 1=white_solid, 2=white_dashed)
        mask_classes = np.zeros_like(mask)
        mask_classes[mask == 128] = 1  # white_solid
        mask_classes[mask == 255] = 2  # white_dashed
        
        # Apply transforms
        if self.is_training:
            # Apply same transform to both image and mask
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            # Simple mask transform (no normalization)
            mask_pil = transforms.ToPILImage()(mask_classes.astype(np.uint8))
            if random.random() < 0.3:  # Match horizontal flip probability
                mask_pil = transforms.functional.hflip(mask_pil)
            mask_classes = np.array(mask_pil)
        else:
            image = self.transform(image)
        
        return image, torch.from_numpy(mask_classes).long()

def calculate_metrics(outputs, targets, num_classes=3):
    """Calculate IoU, Precision, Recall, F1 for each class"""
    preds = torch.argmax(outputs, dim=1)
    
    metrics = {}
    ious = []
    
    for class_id in range(num_classes):
        pred_mask = (preds == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
        
        # Precision, Recall, F1
        if pred_mask.sum() > 0:
            precision = (intersection / pred_mask.float().sum()).item()
        else:
            precision = 1.0 if intersection == 0 else 0.0
            
        if target_mask.sum() > 0:
            recall = (intersection / target_mask.float().sum()).item()
        else:
            recall = 1.0 if intersection == 0 else 0.0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        class_names = ['background', 'white_solid', 'white_dashed']
        metrics[class_names[class_id]] = {
            'iou': iou * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    overall_miou = sum(ious) / len(ious) * 100
    lane_classes_miou = sum(ious[1:]) / len(ious[1:]) * 100
    lane_classes_f1 = sum([metrics['white_solid']['f1'], metrics['white_dashed']['f1']]) / 2
    balanced_score = (overall_miou * 0.4 + lane_classes_miou * 0.4 + lane_classes_f1 * 0.2)
    
    return overall_miou, lane_classes_miou, lane_classes_f1, balanced_score, metrics

def save_checkpoint(model, epoch, miou, balanced_score, is_best=False, is_swa=False):
    """Save model checkpoint with automatic best model detection"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_best:
        # Create backup directory for new best
        backup_dir = Path(f'model_backups/finetune_epoch{epoch}_best_{timestamp}')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_name = f'finetune_best_model_epoch{epoch}_{miou:.1f}miou.pth'
        torch.save(model.state_dict(), backup_dir / model_name)
        
        # Save performance record
        record = {
            'epoch': epoch,
            'overall_miou': miou,
            'balanced_score': balanced_score,
            'timestamp': timestamp,
            'training_type': 'fine_tuning',
            'base_model': 'epoch50_85.1_miou',
            'status': 'NEW BEST'
        }
        
        with open(backup_dir / f'finetune_epoch{epoch}_best_record.json', 'w') as f:
            json.dump(record, f, indent=2)
        
        print(f"✅ NEW BEST saved: {backup_dir / model_name}")
        print(f"📊 Performance: {miou:.1f}% mIoU, {balanced_score:.1f}% Balanced")
    
    elif epoch % 5 == 0:  # Save every 5 epochs as backup
        backup_dir = Path('work_dirs')
        backup_dir.mkdir(exist_ok=True)
        
        suffix = "_swa" if is_swa else ""
        model_name = f'finetune_epoch{epoch}{suffix}_backup.pth'
        torch.save(model.state_dict(), backup_dir / model_name)
        print(f"💾 Backup saved: {backup_dir / model_name}")

def main():
    print("=" * 60)
    print("FINE-TUNING PREMIUM MODEL - Option 3")
    print("Starting from 85.1% mIoU Epoch 50 checkpoint")
    print("Target: 85.5-86.0% mIoU with conservative improvements")
    print("=" * 60)
    print()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()
    
    # Fine-tuning configuration
    print("FINE-TUNING CONFIGURATION:")
    print("- Base model: Epoch 50 (85.1% mIoU)")
    print("- Learning rate: 1e-5 (conservative)")
    print("- Additional epochs: 30 (51-80)")
    print("- SWA: Last 10 epochs (71-80)")
    print("- Gentle augmentations + regularization")
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FineTuningAELDataset('data/train_data.json', is_training=True)
    val_dataset = FineTuningAELDataset('data/val_data.json', is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                             num_workers=4, pin_memory=True)  # Smaller batch for fine-tuning
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Fine-tuning dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print()
    
    # Initialize model
    model = FineTuningPremiumUNet(num_classes=3, dropout_rate=0.2).to(device)
    
    # Load the best checkpoint (Epoch 50 - 85.1% mIoU)
    print("Loading Epoch 50 checkpoint (85.1% mIoU)...")
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model state dict if it's wrapped
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"✅ Best mIoU in checkpoint: {checkpoint.get('best_miou', 'unknown'):.1f}%")
        else:
            state_dict = checkpoint
            print(f"✅ Loaded direct state dict")
        
        # Load with strict=False to handle architecture differences
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Model weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("Attempting to load compatible layers only...")
            
            # Load compatible layers manually
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.size() == model_dict[k].size()}
            
            print(f"✅ Loading {len(pretrained_dict)} compatible layers out of {len(state_dict)} total")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure the Epoch 50 model is available!")
        return
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Loss function with gentler settings
    class_weights = [0.1, 5.0, 5.0]  # Keep proven weights
    criterion = EnhancedDiceFocalLoss(
        alpha=1, gamma=1.5,  # Reduced gamma for fine-tuning
        dice_weight=0.65, focal_weight=0.35,  # Slightly adjusted
        class_weights=class_weights,
        label_smoothing=0.05  # Reduced label smoothing
    ).to(device)
    
    print("Fine-tuning Loss Configuration:")
    print(f"  Dice weight: 0.65, Focal weight: 0.35")
    print(f"  + Edge loss (5%) + Smoothness (2%)")
    print(f"  Class weights: {class_weights}")
    print()
    
    # Optimizer with very low learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Cosine scheduler for fine-tuning (small range)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-6)
    
    # SWA setup (for last 10 epochs)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-6)
    
    print(f"Optimizer: AdamW with LR=1e-5")
    print(f"Scheduler: Cosine annealing (1e-5 -> 5e-6)")
    print(f"SWA: Last 10 epochs with LR=5e-6")
    print()
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training tracking
    best_miou = 85.1  # Start from known best
    best_balanced_score = 77.4
    start_time = time.time()
    
    print("Starting fine-tuning from Epoch 51...")
    print("Auto-saving any mIoU improvement > 0.05%")
    print()
    
    # Fine-tuning loop (Epochs 51-80)
    for epoch in range(51, 81):
        start_epoch_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice_loss = 0.0
        train_focal_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/80 - Fine-tuning")
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss, dice_loss, focal_loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gentler clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_dice_loss += dice_loss
            train_focal_loss += focal_loss
            
            # Update progress
            train_pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Dice': f'{dice_loss:.4f}',
                'Focal': f'{focal_loss:.4f}'
            })
        
        # Update scheduler
        if epoch >= 71:  # SWA phase
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_dice_loss = train_dice_loss / len(train_loader)
        avg_focal_loss = train_focal_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                
                val_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(masks.cpu())
        
        # Calculate metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        overall_miou, lane_miou, lane_f1, balanced_score, class_metrics = calculate_metrics(
            all_outputs, all_targets
        )
        
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_epoch_time
        
        # Display results
        print(f"Epoch {epoch}/80:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Dice: {avg_dice_loss:.4f}, Focal: {avg_focal_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Overall mIoU: {overall_miou:.1f}%")
        print(f"  Lane Classes mIoU: {lane_miou:.1f}%")
        print(f"  Lane Classes F1: {lane_f1:.1f}%")
        print(f"  Balanced Score: {balanced_score:.1f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Detailed Per-class Metrics:")
        for class_name, metrics in class_metrics.items():
            print(f"    {class_name}: IoU {metrics['iou']:.1f}%, "
                  f"Prec {metrics['precision']:.1f}%, "
                  f"Rec {metrics['recall']:.1f}%, "
                  f"F1 {metrics['f1']:.1f}%")
        print()
        
        # Check for new best (improvement > 0.05%)
        improvement = overall_miou - best_miou
        if improvement > 0.05:
            best_miou = overall_miou
            best_balanced_score = balanced_score
            save_checkpoint(model, epoch, overall_miou, balanced_score, is_best=True)
            print(f"🏆 NEW BEST: {overall_miou:.1f}% mIoU (+{improvement:.2f}%)")
        
        # Regular backup every 5 epochs
        save_checkpoint(model, epoch, overall_miou, balanced_score, is_best=False)
        
        # Success indicators
        if overall_miou >= 85.5:
            print(f"🎯 TARGET REACHED: {overall_miou:.1f}% mIoU >= 85.5%")
        if balanced_score >= 78.0:
            print(f"✨ EXCELLENT BALANCE: {balanced_score:.1f}% >= 78%")
        
        print(f"  Time: {epoch_time:.1f}s")
        print("-" * 60)
    
    # Final SWA model evaluation
    if epoch >= 71:
        print("\n" + "=" * 60)
        print("EVALUATING SWA MODEL (Stochastic Weight Averaging)")
        print("=" * 60)
        
        # Update SWA batch normalization
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Evaluate SWA model
        swa_model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="SWA Validation"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = swa_model(images)
                    loss, _, _ = criterion(outputs, masks)
                
                val_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(masks.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        swa_miou, swa_lane_miou, swa_lane_f1, swa_balanced, swa_metrics = calculate_metrics(
            all_outputs, all_targets
        )
        
        print(f"SWA Model Results:")
        print(f"  Overall mIoU: {swa_miou:.1f}%")
        print(f"  Balanced Score: {swa_balanced:.1f}%")
        print(f"  Lane Classes mIoU: {swa_lane_miou:.1f}%")
        
        # Save SWA model if better
        if swa_miou > best_miou:
            save_checkpoint(swa_model.module, 80, swa_miou, swa_balanced, is_best=True, is_swa=True)
            print(f"🏆 SWA BEST: {swa_miou:.1f}% mIoU")
    
    total_time = time.time() - start_time
    print(f"\nFine-tuning completed in {total_time / 3600:.1f} hours")
    print(f"Best mIoU achieved: {best_miou:.1f}% (from 85.1% baseline)")
    print(f"Improvement: +{best_miou - 85.1:.2f}%")
    
    if best_miou >= 85.5:
        print("🎯 SUCCESS: Target 85.5%+ achieved!")
    if best_miou >= 86.0:
        print("🚀 EXCELLENT: 86%+ breakthrough achieved!")

if __name__ == "__main__":
    main()