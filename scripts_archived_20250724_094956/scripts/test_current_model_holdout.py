#!/usr/bin/env python3
"""
Test Current Model on Holdout Test Set
======================================

Test our current 79.4% mIoU model on the fresh 1,564 holdout test samples
that were never used during training or validation.

This will give us the TRUE performance of our current model.
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
from premium_gpu_train import PremiumLaneNet, PremiumDataset, calculate_detailed_metrics

class HoldoutTestDataset(PremiumDataset):
    """Dataset class for holdout test set"""
    
    def __init__(self, img_dir, ann_dir):
        super().__init__(img_dir, ann_dir, mode='val')  # Use val mode (no augmentation)

def test_current_model_on_holdout():
    """Test current best model on holdout test set"""
    
    print("TESTING CURRENT MODEL ON HOLDOUT TEST SET")
    print("=" * 55)
    print("Model: Current best (79.4% mIoU on validation)")
    print("Test set: 1,564 fresh samples never seen during training")
    print("=" * 55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load current best model
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    if not Path(model_path).exists():
        print("ERROR: Current model not found!")
        return
    
    print(f"\nLoading current model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    reported_miou = checkpoint.get('best_miou', 0)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Model loaded: Epoch {epoch}, Reported mIoU {reported_miou*100:.1f}%")
    
    # Load holdout test dataset
    print(f"\nLoading holdout test dataset...")
    test_dataset = HoldoutTestDataset('data/full_ael_mmseg/img_dir/test', 'data/full_ael_mmseg/ann_dir/test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Holdout test samples: {len(test_dataset)}")
    
    # Run inference
    print(f"\nRunning inference on holdout test set...")
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for each sample in batch
            for i in range(predictions.size(0)):
                ious, precs, recalls, f1s = calculate_detailed_metrics(predictions[i].cpu(), masks[i].cpu())
                all_ious.append(ious)
                all_precisions.append(precs)
                all_recalls.append(recalls)
                all_f1s.append(f1s)
            
            # Progress update
            if batch_idx % 25 == 0:
                processed = batch_idx * images.size(0)
                print(f"  Processed {processed}/{len(test_dataset)} samples...")
    
    test_time = time.time() - start_time
    print(f"Inference completed in {test_time:.1f}s")
    
    # Calculate comprehensive metrics
    print(f"\nCalculating final metrics...")
    
    all_ious = np.array(all_ious)
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_f1s = np.array(all_f1s)
    
    # Mean metrics across all samples
    mean_ious = np.mean(all_ious, axis=0)
    mean_precisions = np.mean(all_precisions, axis=0)
    mean_recalls = np.mean(all_recalls, axis=0)
    mean_f1s = np.mean(all_f1s, axis=0)
    
    # Overall metrics
    overall_miou = np.mean(mean_ious)
    lane_miou = np.mean(mean_ious[1:])  # Exclude background
    overall_f1 = np.mean(mean_f1s)
    
    # Results
    print(f"\n" + "="*55)
    print("HOLDOUT TEST SET RESULTS")
    print("="*55)
    
    print(f"Test samples: {len(all_ious)}")
    print(f"Overall mIoU: {overall_miou*100:.1f}%")
    print(f"Lane mIoU: {lane_miou*100:.1f}%")
    print(f"Overall F1: {overall_f1*100:.1f}%")
    
    class_names = ['Background', 'White Solid', 'White Dashed']
    print(f"\nPer-class performance:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    IoU: {mean_ious[i]*100:.1f}%")
        print(f"    Precision: {mean_precisions[i]*100:.1f}%")
        print(f"    Recall: {mean_recalls[i]*100:.1f}%")
        print(f"    F1: {mean_f1s[i]*100:.1f}%")
    
    # Compare with validation performance
    validation_miou = 79.4  # Our previous validation result
    test_miou = overall_miou * 100
    generalization_gap = validation_miou - test_miou
    
    print(f"\nGENERALIZATION ANALYSIS:")
    print(f"  Validation mIoU: {validation_miou:.1f}%")
    print(f"  Test mIoU: {test_miou:.1f}%")
    print(f"  Generalization gap: {generalization_gap:.1f}%")
    
    if abs(generalization_gap) < 2:
        print(f"  Status: EXCELLENT generalization (gap < 2%)")
    elif abs(generalization_gap) < 5:
        print(f"  Status: GOOD generalization (gap < 5%)")
    elif generalization_gap > 5:
        print(f"  Status: OVERFITTING detected (val >> test)")
    else:
        print(f"  Status: Potential validation issues (test >> val)")
    
    # Performance assessment
    print(f"\nPERFORMANCE ASSESSMENT:")
    if test_miou >= 85:
        print(f"  OUTSTANDING: {test_miou:.1f}% >= 85% target!")
    elif test_miou >= 80:
        print(f"  EXCELLENT: {test_miou:.1f}% >= 80% target")
    elif test_miou >= 75:
        print(f"  GOOD: {test_miou:.1f}% >= 75%")
    else:
        print(f"  NEEDS IMPROVEMENT: {test_miou:.1f}% < 75%")
    
    # Lane detection analysis
    lane_detection_avg = np.mean(mean_ious[1:])
    print(f"\nLANE DETECTION ANALYSIS:")
    print(f"  Average lane IoU: {lane_detection_avg*100:.1f}%")
    
    if lane_detection_avg >= 0.7:
        print(f"  Lane detection: EXCELLENT (>70%)")
    elif lane_detection_avg >= 0.6:
        print(f"  Lane detection: GOOD (>60%)")
    elif lane_detection_avg >= 0.5:
        print(f"  Lane detection: ADEQUATE (>50%)")
    else:
        print(f"  Lane detection: NEEDS IMPROVEMENT (<50%)")
    
    print("="*55)
    
    return {
        'test_miou': overall_miou,
        'validation_miou': validation_miou / 100,
        'generalization_gap': generalization_gap,
        'lane_miou': lane_miou,
        'class_ious': mean_ious.tolist(),
        'test_samples': len(all_ious)
    }

def main():
    """Run holdout test evaluation"""
    results = test_current_model_on_holdout()
    
    print(f"\nCONCLUSION:")
    
    test_miou = results['test_miou'] * 100
    gap = results['generalization_gap']
    
    if test_miou >= 80 and abs(gap) < 5:
        print(f"SUCCESS: Model performs excellently ({test_miou:.1f}%) with good generalization!")
        print(f"RECOMMENDATION: Deploy current model - it's already production-ready.")
    elif test_miou >= 75:
        print(f"GOOD: Model shows solid performance ({test_miou:.1f}%).")
        print(f"RECOMMENDATION: Current model is usable, consider minor improvements.")
    else:
        print(f"IMPROVEMENT NEEDED: Model shows {test_miou:.1f}% on test set.")
        print(f"RECOMMENDATION: Further training or methodology improvements needed.")
    
    return results

if __name__ == "__main__":
    main()