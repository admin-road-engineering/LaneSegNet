#!/usr/bin/env python3
"""
Quick Model Test - Simple evaluation without Unicode issues
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

def quick_evaluation():
    """Quick evaluation of current model"""
    print("=" * 60)
    print("QUICK MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint_path = 'work_dirs/premium_gpu_best_model.pth'
    if not Path(checkpoint_path).exists():
        print("ERROR: No model checkpoint found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create and load model
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: Epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Baseline mIoU: {checkpoint.get('best_miou', 0)*100:.1f}%")
    print()
    
    # Load validation dataset
    val_dataset = PremiumDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Validation samples: {len(val_dataset)}")
    print("Running quick evaluation on 100 samples...")
    print()
    
    # Evaluation
    all_predictions = []
    all_targets = []
    sample_count = 0
    max_samples = 100
    
    with torch.no_grad():
        for images, masks in val_loader:
            if sample_count >= max_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            sample_count += images.size(0)
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Per-class IoU
    class_names = ['background', 'white_solid', 'white_dashed']
    ious = []
    
    print("PER-CLASS RESULTS:")
    for class_id, class_name in enumerate(class_names):
        pred_class = (all_predictions == class_id)
        target_class = (all_targets == class_id)
        
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        
        print(f"  {class_name}: {iou*100:.1f}% IoU")
    
    # Overall metrics
    overall_miou = np.mean(ious)
    lane_classes_miou = np.mean(ious[1:])  # Only lane classes
    
    print()
    print("OVERALL RESULTS:")
    print(f"  Overall mIoU: {overall_miou*100:.1f}%")
    print(f"  Lane mIoU: {lane_classes_miou*100:.1f}%")
    print(f"  Background IoU: {ious[0]*100:.1f}%")
    print()
    
    # Class distribution analysis
    print("CLASS DISTRIBUTION:")
    for class_id, class_name in enumerate(class_names):
        pred_pixels = (all_predictions == class_id).sum()
        target_pixels = (all_targets == class_id).sum()
        total_pixels = all_targets.size
        
        pred_percent = (pred_pixels / total_pixels) * 100
        target_percent = (target_pixels / total_pixels) * 100
        
        print(f"  {class_name}:")
        print(f"    Target: {target_percent:.1f}%, Predicted: {pred_percent:.1f}%")
    
    print()
    print("=" * 60)
    
    return overall_miou, lane_classes_miou

if __name__ == "__main__":
    quick_evaluation()