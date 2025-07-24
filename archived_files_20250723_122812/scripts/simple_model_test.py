#!/usr/bin/env python3
"""
Simple Model Test - Minimal samples with frequent progress updates
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

def simple_test():
    print("SIMPLE MODEL TEST - 50 SAMPLES ONLY")
    print("=" * 45)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test just current model with very few samples
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Very small dataset for testing
    print("Loading dataset...")
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)  # No multiprocessing
    print(f"Dataset ready: {len(val_dataset)} total samples")
    
    # Test on just 50 samples
    print("\\nStarting evaluation on 50 samples...")
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if sample_count >= 50:
                break
            
            print(f"Batch {batch_idx + 1}: Processing {images.size(0)} samples...")
            
            images = images.to(device)
            masks = masks.to(device)
            
            print(f"  Running inference...")
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            sample_count += images.size(0)
            elapsed = time.time() - start_time
            
            print(f"  Completed! Total samples: {sample_count}/50, Time: {elapsed:.1f}s")
            
            if sample_count >= 50:
                break
    
    print(f"\\nEvaluation completed: {sample_count} samples in {time.time() - start_time:.1f}s")
    
    # Quick IoU calculation
    print("Calculating results...")
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate IoU for each class
    ious = []
    class_names = ['background', 'white_solid', 'white_dashed']
    
    for class_id, class_name in enumerate(class_names):
        pred_class = (all_predictions == class_id)
        target_class = (all_targets == class_id)
        
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        print(f"  {class_name}: {iou*100:.1f}% IoU")
    
    overall_miou = np.mean(ious)
    
    print(f"\\nRESULT:")
    print(f"  Samples tested: {sample_count}")
    print(f"  Overall mIoU: {overall_miou*100:.1f}%")
    print(f"  Reported mIoU: 85.1%")
    print(f"  Difference: {85.1 - overall_miou*100:.1f}%")
    
    print("\\nSIMPLE TEST COMPLETED!")

if __name__ == "__main__":
    simple_test()