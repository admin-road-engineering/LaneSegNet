#!/usr/bin/env python3
"""
Quick Model Comparison - Fast evaluation of top backup models
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

def quick_model_test():
    print("=" * 60)
    print("QUICK MODEL COMPARISON")
    print("Testing key backup models (100 samples each)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load validation dataset
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Key models to test (most promising ones)
    key_models = [
        'work_dirs/premium_gpu_best_model.pth',  # Current "best"
        'model_backups/epoch50_final_masterpiece_20250722_194650/premium_gpu_best_model_EPOCH50_FINAL_MASTERPIECE.pth',
        'model_backups/epoch46_85_percent_milestone_20250722_191631/premium_gpu_best_model_EPOCH46_85PERCENT.pth',
        'model_backups/epoch48_sustained_excellence_20250722_192649/premium_gpu_best_model_EPOCH48_SUSTAINED.pth',
    ]
    
    results = []
    
    for model_path in key_models:
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"SKIP: {model_file.name} (not found)")
            continue
        
        print(f"\\nTesting: {model_file.name}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            reported_miou = checkpoint.get('best_miou', 0)
            epoch = checkpoint.get('epoch', 'unknown')
            
            # Quick evaluation (100 samples)
            all_predictions = []
            all_targets = []
            sample_count = 0
            
            start_time = time.time()
            
            with torch.no_grad():
                for images, masks in val_loader:
                    if sample_count >= 100:
                        break
                    
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(masks.cpu().numpy())
                    
                    sample_count += images.size(0)
            
            eval_time = time.time() - start_time
            
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
            
            overall_miou = np.mean(ious)
            lane_miou = np.mean(ious[1:])
            
            result = {
                'model_path': model_path,
                'name': model_file.name,
                'epoch': epoch,
                'reported_miou': reported_miou,
                'actual_miou': overall_miou,
                'lane_miou': lane_miou,
                'background_iou': ious[0],
                'white_solid_iou': ious[1],
                'white_dashed_iou': ious[2],
                'samples': sample_count,
                'eval_time': eval_time
            }
            
            results.append(result)
            
            print(f"  Reported: {reported_miou*100:.1f}%, Actual: {overall_miou*100:.1f}%, Lane: {lane_miou*100:.1f}%")
            print(f"  Classes: BG={ious[0]*100:.1f}%, Solid={ious[1]*100:.1f}%, Dashed={ious[2]*100:.1f}%")
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Results summary
    print(f"\\n" + "=" * 60)
    print("QUICK COMPARISON RESULTS")
    print("=" * 60)
    
    if results:
        # Sort by actual mIoU
        results.sort(key=lambda x: x['actual_miou'], reverse=True)
        
        print(f"{'Model':<25} {'Reported':<9} {'Actual':<8} {'Lane':<7} {'Diff':<6}")
        print("-" * 60)
        
        for result in results:
            name = result['name'][:23] + '..' if len(result['name']) > 25 else result['name']
            reported = result['reported_miou'] * 100
            actual = result['actual_miou'] * 100
            lane = result['lane_miou'] * 100
            diff = reported - actual
            
            print(f"{name:<25} {reported:<9.1f} {actual:<8.1f} {lane:<7.1f} {diff:<6.1f}")
        
        best = results[0]
        print(f"\\nBEST MODEL: {best['name']}")
        print(f"  Actual mIoU: {best['actual_miou']*100:.1f}%")
        print(f"  Lane mIoU: {best['lane_miou']*100:.1f}%")
        print(f"  Model path: {best['model_path']}")
        
    print("=" * 60)

if __name__ == "__main__":
    quick_model_test()