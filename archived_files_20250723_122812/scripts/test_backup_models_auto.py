#!/usr/bin/env python3
"""
Automated Backup Model Testing
=============================

Non-interactive version that tests all backup models automatically
with 100 samples to find the true best performer.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import json

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

def find_backup_models():
    """Find all available backup model files"""
    print("Searching for backup models...")
    
    # Search locations
    search_paths = [
        'model_backups',
        'work_dirs',
        '.',
    ]
    
    model_files = []
    
    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue
            
        # Find .pth files recursively
        for model_file in path.rglob('*.pth'):
            if 'premium' in model_file.name.lower() or 'best' in model_file.name.lower():
                model_files.append(model_file)
    
    # Remove duplicates and sort
    unique_models = list(set(model_files))
    unique_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Newest first
    
    print(f"Found {len(unique_models)} backup models:")
    for i, model_file in enumerate(unique_models, 1):
        print(f"  {i}. {model_file}")
    
    return unique_models

def test_single_model(model_path, test_samples=100):
    """Test a single model and return performance metrics"""
    print(f"\nTesting: {model_path}")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract metadata
        epoch = checkpoint.get('epoch', 'unknown')
        reported_miou = checkpoint.get('best_miou', 0)
        
        print(f"  Reported Epoch: {epoch}")
        print(f"  Reported mIoU: {reported_miou*100:.1f}%")
        
        # Load model
        model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("  Model loaded successfully")
        
        # Load validation data
        val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Test on specified number of samples
        print(f"  Testing on {test_samples} samples...")
        
        all_predictions = []
        all_targets = []
        sample_count = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if sample_count >= test_samples:
                    break
                
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                sample_count += images.size(0)
                
                if batch_idx % 5 == 0:
                    print(f"    Processed {sample_count}/{test_samples} samples")
        
        test_time = time.time() - start_time
        print(f"  Testing completed in {test_time:.1f}s")
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Per-class IoU
        class_names = ['background', 'white_solid', 'white_dashed']
        ious = []
        
        for class_id, class_name in enumerate(class_names):
            pred_class = (all_predictions == class_id)
            target_class = (all_targets == class_id)
            
            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
            print(f"    {class_name}: {iou*100:.1f}% IoU")
        
        overall_miou = np.mean(ious)
        lane_miou = np.mean(ious[1:])  # Exclude background
        
        # Coverage statistics
        pred_lane_coverage = ((all_predictions == 1) | (all_predictions == 2)).mean()
        gt_lane_coverage = ((all_targets == 1) | (all_targets == 2)).mean()
        
        print(f"  Overall mIoU: {overall_miou*100:.1f}%")
        print(f"  Lane mIoU: {lane_miou*100:.1f}%")
        print(f"  Predicted lane coverage: {pred_lane_coverage*100:.2f}%")
        print(f"  Ground truth lane coverage: {gt_lane_coverage*100:.2f}%")
        
        # Calculate discrepancy
        discrepancy = abs(reported_miou - overall_miou) * 100
        print(f"  Performance discrepancy: {discrepancy:.1f}%")
        
        result = {
            'model_path': str(model_path),
            'model_name': model_path.name,
            'epoch': epoch,
            'reported_miou': reported_miou,
            'actual_miou': overall_miou,
            'lane_miou': lane_miou,
            'discrepancy': discrepancy,
            'class_ious': ious,
            'pred_lane_coverage': pred_lane_coverage,
            'gt_lane_coverage': gt_lane_coverage,
            'test_time': test_time,
            'samples_tested': sample_count,
            'loading_success': True
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'model_path': str(model_path),
            'model_name': model_path.name,
            'loading_success': False,
            'error': str(e),
            'actual_miou': 0,
            'discrepancy': float('inf')
        }

def main():
    print("AUTOMATED BACKUP MODEL TESTING")
    print("=" * 60)
    print("Testing each model on 100 validation samples")
    print("=" * 60)
    
    # Find all models
    backup_models = find_backup_models()
    
    if not backup_models:
        print("No backup models found!")
        return
    
    print(f"\nTesting {len(backup_models)} models...")
    
    results = []
    
    # Test each model
    for i, model_path in enumerate(backup_models, 1):
        print(f"\n[{i}/{len(backup_models)}] Testing {model_path.name}")
        result = test_single_model(model_path, test_samples=100)
        results.append(result)
    
    # Generate comparison report
    print("\n" + "="*60)
    print("BACKUP MODEL COMPARISON REPORT")
    print("="*60)
    
    # Filter successful tests
    successful_results = [r for r in results if r.get('loading_success', False)]
    failed_results = [r for r in results if not r.get('loading_success', False)]
    
    print(f"\nSUMMARY:")
    print(f"  Total models tested: {len(results)}")
    print(f"  Successful tests: {len(successful_results)}")
    print(f"  Failed tests: {len(failed_results)}")
    
    if failed_results:
        print(f"\nFAILED MODELS:")
        for result in failed_results:
            print(f"  {result['model_name']}: {result.get('error', 'Unknown error')}")
    
    if not successful_results:
        print("\nNo models tested successfully!")
        return
    
    # Sort by actual performance
    successful_results.sort(key=lambda x: x['actual_miou'], reverse=True)
    
    print(f"\nPERFORMANCE RANKING (by actual mIoU):")
    print("-" * 40)
    for i, result in enumerate(successful_results, 1):
        reported = result['reported_miou'] * 100
        actual = result['actual_miou'] * 100
        discrepancy = result['discrepancy']
        
        print(f"{i}. {result['model_name']}")
        print(f"   Reported: {reported:.1f}%  |  Actual: {actual:.1f}%  |  Gap: {discrepancy:.1f}%")
        print(f"   Lane mIoU: {result['lane_miou']*100:.1f}%  |  Lane coverage: {result['pred_lane_coverage']*100:.2f}%")
    
    # Identify best model
    best_model = successful_results[0]
    print(f"\nBEST PERFORMING MODEL:")
    print(f"  Model: {best_model['model_name']}")
    print(f"  Actual mIoU: {best_model['actual_miou']*100:.1f}%")
    print(f"  Lane detection: {best_model['pred_lane_coverage']*100:.2f}% coverage")
    
    # Save detailed report
    report_path = f"backup_model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_samples': 100,
            'results': results,
            'best_model': best_model['model_path'] if successful_results else None,
            'summary': {
                'total_tested': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results)
            }
        }, f, indent=2, default=str)
    
    print(f"\nDetailed report saved: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()