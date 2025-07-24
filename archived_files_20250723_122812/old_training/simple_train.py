#!/usr/bin/env python3
"""
Simple training script for Phase 3.2 using direct PyTorch approach
Bypasses MMCV extension issues
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import time

class SimpleAELDataset(Dataset):
    """Simple AEL dataset for 4-class lane detection."""
    
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get list of image files
        self.images = list(self.img_dir.glob("*.jpg"))
        print(f"Found {len(self.images)} images in {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        # Load and resize image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        # Load and resize mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Ensure mask values are in [0, 3] for 4 classes
        mask = np.clip(mask, 0, 3)
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class SimpleLaneNet(nn.Module):
    """Simple lane detection network for baseline."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

def calculate_iou(pred, target, num_classes=4):
    """Calculate IoU for each class."""
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
            iou = intersection / union
        
        ious.append(iou)
    
    return ious

def train_model():
    """Train the simple lane detection model."""
    print("Phase 3.2: Simple Lane Detection Training")
    print("Target: Establish baseline performance for 4-class detection")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    train_img_dir = "data/ael_mmseg/img_dir/train"
    train_mask_dir = "data/ael_mmseg/ann_dir/train"
    val_img_dir = "data/ael_mmseg/img_dir/val"
    val_mask_dir = "data/ael_mmseg/ann_dir/val"
    
    # Datasets
    train_dataset = SimpleAELDataset(train_img_dir, train_mask_dir)
    val_dataset = SimpleAELDataset(val_img_dir, val_mask_dir)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Model
    model = SimpleLaneNet(num_classes=4).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Training loop
    num_epochs = 25
    best_miou = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            train_bar.set_postfix(loss=loss.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_ious = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
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
        print(f"  Val mIoU: {miou:.4f}")
        print(f"  Class IoUs: {[f'{iou:.3f}' for iou in mean_ious]}")
        
        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "work_dirs/best_model.pth")
            print(f"  New best mIoU: {best_miou:.4f}")
        
        print("-" * 50)
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    
    print(f"\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Total time: {int(hours)}h {int(minutes)}m")
    
    # Save final results
    results = {
        "best_miou": best_miou,
        "training_time_hours": elapsed_time / 3600,
        "num_epochs": num_epochs,
        "num_classes": 4
    }
    
    Path("work_dirs").mkdir(exist_ok=True)
    with open("work_dirs/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_miou

if __name__ == "__main__":
    # Create work directory
    Path("work_dirs").mkdir(exist_ok=True)
    
    # Run training
    final_miou = train_model()
    
    print(f"\nPhase 3.2 Baseline Complete!")
    print(f"Final mIoU: {final_miou:.1%}")
    
    if final_miou >= 0.65:
        print("✓ Baseline target (65%+ mIoU) achieved!")
    else:
        print("⚠ Below baseline target - may need more training")