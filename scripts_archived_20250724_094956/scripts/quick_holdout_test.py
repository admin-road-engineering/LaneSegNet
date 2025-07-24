#!/usr/bin/env python3
"""
Quick Holdout Test
=================

Quick test of current model on 100 random holdout samples
to get fast assessment of true performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import random

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

def quick_holdout_test():
    """Quick test on 100 random holdout samples"""
    
    print("EXPANDED HOLDOUT TEST")
    print("=" * 35)
    print("Testing current model on 300 random holdout samples")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load current model
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    reported_miou = checkpoint.get('best_miou', 0)
    print(f"Model: Reported {reported_miou*100:.1f}% mIoU")
    
    # Load holdout test dataset
    test_dataset = PremiumDataset('data/full_ael_mmseg/img_dir/test', 'data/full_ael_mmseg/ann_dir/test', mode='val')
    
    # Sample 300 random indices for more reliable results
    total_samples = len(test_dataset)
    test_indices = random.sample(range(total_samples), min(300, total_samples))
    
    print(f"Testing {len(test_indices)} random samples from {total_samples} holdout samples")
    
    # Quick IoU calculation
    def calculate_iou_simple(pred, target):
        ious = []
        for class_id in range(3):
            pred_class = (pred == class_id)
            target_class = (target == class_id)
            intersection = (pred_class & target_class).sum()
            union = (pred_class | target_class).sum()
            iou = intersection.float() / union.float() if union > 0 else 0
            ious.append(iou.item() if hasattr(iou, 'item') else float(iou))
        return ious
    
    all_ious = []
    start_time = time.time()
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            image, mask = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.to(device)
            
            outputs = model(image)
            prediction = torch.argmax(outputs, dim=1)[0]
            
            ious = calculate_iou_simple(prediction, mask)
            all_ious.append(ious)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_indices)} samples")
    
    test_time = time.time() - start_time
    
    # Calculate results
    all_ious = np.array(all_ious)
    mean_ious = np.mean(all_ious, axis=0)
    overall_miou = np.mean(mean_ious)
    lane_miou = np.mean(mean_ious[1:])
    
    print(f"\nEXPANDED HOLDOUT RESULTS:")
    
    # Also analyze distribution to see if there are ANY lane detections
    lane_predictions = all_ious[:, 1:].sum(axis=1)  # Sum of white_solid + white_dashed
    samples_with_lanes = (lane_predictions > 0).sum()
    print(f"  Samples with ANY lane detection: {samples_with_lanes}/{len(all_ious)}")
    print(f"  Samples tested: {len(all_ious)}")
    print(f"  Overall mIoU: {overall_miou*100:.1f}%")
    print(f"  Lane mIoU: {lane_miou*100:.1f}%")
    print(f"  Background: {mean_ious[0]*100:.1f}%")
    print(f"  White Solid: {mean_ious[1]*100:.1f}%")
    print(f"  White Dashed: {mean_ious[2]*100:.1f}%")
    print(f"  Test time: {test_time:.1f}s")
    
    # Compare with validation
    validation_miou = 79.4
    test_miou = overall_miou * 100
    gap = validation_miou - test_miou
    
    print(f"\nCOMPARISON:")
    print(f"  Validation: {validation_miou:.1f}%")
    print(f"  Holdout test: {test_miou:.1f}%")
    print(f"  Gap: {gap:.1f}%")
    
    if abs(gap) < 3:
        print(f"  EXCELLENT generalization!")
    elif abs(gap) < 5:
        print(f"  GOOD generalization")
    else:
        print(f"  Significant generalization gap")
    
    return overall_miou

if __name__ == "__main__":
    result = quick_holdout_test()