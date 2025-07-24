#!/usr/bin/env python3
"""
Test Top 3 Models - Quick validation of our best candidates
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

def test_top_models():
    print("TOP 3 MODEL VALIDATION")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Top 3 models based on reported performance
    top_models = [
        {
            'name': 'Current Best (work_dirs)',
            'path': 'work_dirs/premium_gpu_best_model.pth',
            'expected': '85.1%'
        },
        {
            'name': 'Epoch 50 Final Masterpiece', 
            'path': 'model_backups/epoch50_final_masterpiece_20250722_194650/premium_gpu_best_model_EPOCH50_FINAL_MASTERPIECE.pth',
            'expected': '85.1%'
        },
        {
            'name': 'Epoch 46 Milestone',
            'path': 'model_backups/epoch46_85_percent_milestone_20250722_191631/premium_gpu_best_model_EPOCH46_85PERCENT.pth', 
            'expected': '85.0%'
        }
    ]
    
    # Load validation data (small subset for speed)
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Testing on 200 validation samples...")
    print()
    
    results = []
    
    for i, model_info in enumerate(top_models):
        print(f"[{i+1}/3] {model_info['name']} (Expected: {model_info['expected']})")
        
        if not Path(model_info['path']).exists():
            print("  SKIP: File not found")
            continue
        
        try:
            # Load model
            checkpoint = torch.load(model_info['path'], map_location='cpu')
            model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            # Quick evaluation
            all_predictions = []
            all_targets = []
            sample_count = 0
            
            with torch.no_grad():
                batch_count = 0
                for images, masks in val_loader:
                    if sample_count >= 200:
                        break
                    
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(masks.cpu().numpy())
                    
                    sample_count += images.size(0)
                    batch_count += 1
                    
                    # Progress update every 5 batches
                    if batch_count % 5 == 0:
                        print(f"    Progress: {sample_count}/200 samples processed...")
                
                print(f"    Completed: {sample_count} samples processed")
            
            # Calculate IoU
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            ious = []
            class_names = ['background', 'white_solid', 'white_dashed']
            
            for class_id in range(3):
                pred_class = (all_predictions == class_id)
                target_class = (all_targets == class_id)
                
                intersection = np.logical_and(pred_class, target_class).sum()
                union = np.logical_or(pred_class, target_class).sum()
                
                iou = intersection / union if union > 0 else 0
                ious.append(iou)
            
            overall_miou = np.mean(ious)
            lane_miou = np.mean(ious[1:])
            
            results.append({
                'name': model_info['name'],
                'path': model_info['path'],
                'expected': model_info['expected'],
                'actual_miou': overall_miou,
                'lane_miou': lane_miou,
                'ious': ious
            })
            
            print(f"  Actual mIoU: {overall_miou*100:.1f}%")
            print(f"  Lane mIoU: {lane_miou*100:.1f}%")
            print(f"  Per-class: BG={ious[0]*100:.1f}%, Solid={ious[1]*100:.1f}%, Dashed={ious[2]*100:.1f}%")
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    # Summary
    if results:
        print("COMPARISON SUMMARY")
        print("=" * 50)
        
        # Sort by actual performance
        results.sort(key=lambda x: x['actual_miou'], reverse=True)
        
        print(f"{'Rank':<2} {'Model':<25} {'Expected':<9} {'Actual':<8} {'Lane':<7}")
        print("-" * 55)
        
        for i, result in enumerate(results):
            name = result['name'][:23] + '..' if len(result['name']) > 25 else result['name']
            print(f"{i+1:<2} {name:<25} {result['expected']:<9} {result['actual_miou']*100:<8.1f} {result['lane_miou']*100:<7.1f}")
        
        winner = results[0]
        print(f"\\nWINNER: {winner['name']}")
        print(f"  Path: {winner['path']}")
        print(f"  Actual mIoU: {winner['actual_miou']*100:.1f}%")
        print(f"  Lane detection: {winner['lane_miou']*100:.1f}%")
        
        return winner['path']
    
    return None

if __name__ == "__main__":
    best_model = test_top_models()