#!/usr/bin/env python3
"""
Test Single Model - Quick test with progress monitoring
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

def test_single_model():
    print("SINGLE MODEL TEST WITH PROGRESS")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test current best model
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    
    if not Path(model_path).exists():
        print("ERROR: Model not found!")
        return
    
    print(f"Testing: {model_path}")
    
    # Load validation data
    print("Loading validation dataset...")
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    print(f"Dataset loaded: {len(val_dataset)} samples")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    reported_miou = checkpoint.get('best_miou', 0)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Model loaded: Epoch {epoch}, Reported mIoU: {reported_miou*100:.1f}%")
    
    # Test on 100 samples with progress
    print(f"\\nStarting evaluation on 100 samples...")
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    batch_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, masks in val_loader:
            if sample_count >= 100:
                break
            
            batch_count += 1
            print(f"Processing batch {batch_count}...")
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            sample_count += images.size(0)
            print(f"  Processed {sample_count} samples so far...")
            
            # Show timing
            elapsed = time.time() - start_time
            print(f"  Elapsed time: {elapsed:.1f}s")
    
    print(f"\\nEvaluation completed: {sample_count} samples in {time.time() - start_time:.1f}s")
    
    # Calculate IoU
    print("Calculating metrics...")
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    class_names = ['background', 'white_solid', 'white_dashed']
    ious = []
    
    for class_id, class_name in enumerate(class_names):
        pred_class = (all_predictions == class_id)
        target_class = (all_targets == class_id)
        
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        print(f"  {class_name}: {iou*100:.1f}% IoU")
    
    overall_miou = np.mean(ious)
    lane_miou = np.mean(ious[1:])
    
    print(f"\\nFINAL RESULTS:")
    print(f"  Overall mIoU: {overall_miou*100:.1f}%")
    print(f"  Lane mIoU: {lane_miou*100:.1f}%")
    print(f"  Reported vs Actual: {reported_miou*100:.1f}% vs {overall_miou*100:.1f}%")
    print(f"  Difference: {(reported_miou - overall_miou)*100:.1f}%")
    
    print("\\nTEST COMPLETED SUCCESSFULLY!")
    print("=" * 40)

if __name__ == "__main__":
    test_single_model()