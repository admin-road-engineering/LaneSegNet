#!/usr/bin/env python3
"""
Test Model on Training Data
===========================

Test our current model on training images to verify if the testing methodology
is flawed or if the model truly has 0% lane detection capability.

If the model shows good performance on training data but 0% on test data,
then it's overfitting. If it shows 0% on both, there's a testing methodology issue.
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

def test_on_training_data():
    """Test model on training data to debug testing methodology"""
    
    print("TESTING MODEL ON TRAINING DATA")
    print("=" * 40)
    print("Purpose: Debug testing methodology")
    print("If 0% lanes here too -> testing bug")
    print("If good lanes here -> overfitting confirmed")
    print("=" * 40)
    
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
    
    # Test on BOTH training and validation datasets
    datasets_to_test = [
        ('Training', 'data/ael_mmseg/img_dir/train', 'data/ael_mmseg/ann_dir/train'),
        ('Validation', 'data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val'),
        ('Holdout Test', 'data/full_ael_mmseg/img_dir/test', 'data/full_ael_mmseg/ann_dir/test')
    ]
    
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
    
    results_summary = {}
    
    for dataset_name, img_dir, ann_dir in datasets_to_test:
        print(f"\n--- TESTING ON {dataset_name.upper()} DATA ---")
        
        # Check if dataset exists
        if not Path(img_dir).exists():
            print(f"Dataset not found: {img_dir}")
            continue
            
        # Load dataset
        try:
            dataset = PremiumDataset(img_dir, ann_dir, mode='val')
            total_samples = len(dataset)
            print(f"Dataset size: {total_samples} samples")
            
            # Sample random images for testing
            test_count = min(100, total_samples)
            test_indices = random.sample(range(total_samples), test_count)
            
            print(f"Testing {test_count} random samples...")
            
            all_ious = []
            samples_with_lanes = 0
            lane_pixel_counts = []
            
            start_time = time.time()
            
            with torch.no_grad():
                for i, idx in enumerate(test_indices):
                    image, mask = dataset[idx]
                    image = image.unsqueeze(0).to(device)
                    mask = mask.to(device)
                    
                    # Run inference
                    outputs = model(image)
                    prediction = torch.argmax(outputs, dim=1)[0]
                    
                    # Calculate IoUs
                    ious = calculate_iou_simple(prediction, mask)
                    all_ious.append(ious)
                    
                    # Count lane pixels predicted
                    lane_pixels = ((prediction == 1) | (prediction == 2)).sum().item()
                    lane_pixel_counts.append(lane_pixels)
                    
                    if lane_pixels > 0:
                        samples_with_lanes += 1
                    
                    # Ground truth lane pixels for comparison
                    gt_lane_pixels = ((mask == 1) | (mask == 2)).sum().item()
                    
                    if (i + 1) % 25 == 0:
                        print(f"  Processed {i + 1}/{test_count} samples")
                        print(f"    Latest: Pred={lane_pixels} lane pixels, GT={gt_lane_pixels} lane pixels")
            
            test_time = time.time() - start_time
            
            # Calculate results
            all_ious = np.array(all_ious)
            mean_ious = np.mean(all_ious, axis=0)
            overall_miou = np.mean(mean_ious)
            lane_miou = np.mean(mean_ious[1:])
            
            # Lane detection statistics
            avg_lane_pixels = np.mean(lane_pixel_counts)
            max_lane_pixels = np.max(lane_pixel_counts)
            
            print(f"\nRESULTS for {dataset_name}:")
            print(f"  Samples tested: {len(all_ious)}")
            print(f"  Overall mIoU: {overall_miou*100:.1f}%")
            print(f"  Lane mIoU: {lane_miou*100:.1f}%")
            print(f"  Background: {mean_ious[0]*100:.1f}%")
            print(f"  White Solid: {mean_ious[1]*100:.1f}%")
            print(f"  White Dashed: {mean_ious[2]*100:.1f}%")
            print(f"  Samples with lane detection: {samples_with_lanes}/{test_count}")
            print(f"  Average lane pixels predicted: {avg_lane_pixels:.1f}")
            print(f"  Max lane pixels predicted: {max_lane_pixels}")
            print(f"  Test time: {test_time:.1f}s")
            
            results_summary[dataset_name] = {
                'overall_miou': overall_miou * 100,
                'lane_miou': lane_miou * 100,
                'samples_with_lanes': samples_with_lanes,
                'total_samples': test_count,
                'avg_lane_pixels': avg_lane_pixels
            }
            
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
            continue
    
    # Analysis and comparison
    print(f"\n" + "="*50)
    print("COMPARATIVE ANALYSIS")
    print("="*50)
    
    for dataset_name, results in results_summary.items():
        print(f"{dataset_name}:")
        print(f"  mIoU: {results['overall_miou']:.1f}%")
        print(f"  Lane detection: {results['samples_with_lanes']}/{results['total_samples']} samples")
        print(f"  Lane pixels: {results['avg_lane_pixels']:.1f} avg")
    
    # Diagnosis
    print(f"\nDIAGNOSIS:")
    
    if len(results_summary) >= 2:
        training_results = results_summary.get('Training', {})
        test_results = results_summary.get('Holdout Test', {})
        
        training_lanes = training_results.get('samples_with_lanes', 0)
        test_lanes = test_results.get('samples_with_lanes', 0)
        
        if training_lanes > 0 and test_lanes == 0:
            print("  OVERFITTING: Model works on training data but fails on test data")
            print("  -> Model memorized training patterns without learning generalizable features")
        elif training_lanes == 0 and test_lanes == 0:
            print("  TESTING BUG: Model fails on both training and test data")
            print("  -> Issue with testing methodology or model loading")
        elif training_lanes > 0 and test_lanes > 0:
            print("  NORMAL: Model shows some performance on both datasets")
            print("  -> May need investigation of performance gap")
        else:
            print("  UNCLEAR: Mixed results require further investigation")
    
    return results_summary

def main():
    """Run comparative testing analysis"""  
    results = test_on_training_data()
    
    print(f"\nCONCLUSION:")
    if not results:
        print("No datasets could be tested - check dataset paths")
    else:
        training_perf = results.get('Training', {}).get('samples_with_lanes', 0)
        test_perf = results.get('Holdout Test', {}).get('samples_with_lanes', 0)
        
        if training_perf == 0:
            print("CRITICAL: Model fails even on training data!")
            print("-> Check model loading, dataset preprocessing, or model architecture")
        elif test_perf == 0:
            print("CONFIRMED: Severe overfitting detected")
            print("-> Model memorized training data without learning lane detection")
        else:
            print("Model shows lane detection capability - investigate performance gaps")
    
    return results

if __name__ == "__main__":
    main()