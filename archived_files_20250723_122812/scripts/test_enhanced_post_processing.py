#!/usr/bin/env python3
"""
Test Enhanced Post-Processing Pipeline
=====================================

Test the enhanced post-processing pipeline to measure performance improvement
from the baseline 78.4% mIoU model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
import time

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset, calculate_detailed_metrics
from enhanced_post_processing import LanePostProcessor

def calculate_iou_simple(pred_mask, gt_mask):
    """Simple IoU calculation for comparison"""
    pred_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
    gt_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
    
    class_names = ['background', 'white_solid', 'white_dashed']
    ious = []
    
    for class_id in range(3):
        pred_class = (pred_np == class_id)
        gt_class = (gt_np == class_id)
        
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    return ious, class_names

def test_enhanced_post_processing():
    """Test enhanced post-processing pipeline"""
    print("ENHANCED POST-PROCESSING EVALUATION")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load('work_dirs/premium_gpu_best_model.pth', map_location='cpu')
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    reported_miou = checkpoint.get('best_miou', 0)
    print(f"Model loaded: Reported mIoU {reported_miou*100:.1f}%")
    
    # Load validation data
    print("Loading validation dataset...")
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize post-processor
    post_processor = LanePostProcessor(min_lane_area=50, kernel_size=3)
    
    # Test on 100 samples
    test_samples = 100
    print(f"Testing on {test_samples} samples...")
    
    raw_results = []
    processed_results = []
    sample_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if sample_count >= test_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            raw_predictions = torch.argmax(outputs, dim=1)
            
            # Process each sample in batch
            for i in range(images.size(0)):
                if sample_count >= test_samples:
                    break
                
                raw_pred = raw_predictions[i]
                gt_mask = masks[i]
                original_image = images[i]
                
                # Apply post-processing
                processed_pred = post_processor.full_post_processing(raw_pred, original_image)
                
                # Calculate IoUs
                raw_ious, class_names = calculate_iou_simple(raw_pred, gt_mask)
                processed_ious, _ = calculate_iou_simple(processed_pred, gt_mask)
                
                raw_results.append(raw_ious)
                processed_results.append(processed_ious)
                
                sample_count += 1
            
            if batch_idx % 5 == 0:
                print(f"  Processed {sample_count}/{test_samples} samples...")
    
    test_time = time.time() - start_time
    print(f"Testing completed in {test_time:.1f}s")
    
    # Calculate statistics
    raw_results = np.array(raw_results)
    processed_results = np.array(processed_results)
    
    raw_mean = np.mean(raw_results, axis=0)
    processed_mean = np.mean(processed_results, axis=0)
    
    raw_overall = np.mean(raw_mean)
    processed_overall = np.mean(processed_mean)
    
    raw_lane = np.mean(raw_mean[1:])  # Exclude background
    processed_lane = np.mean(processed_mean[1:])  # Exclude background
    
    # Results summary
    print(f"\nRESULTS SUMMARY:")
    print(f"=" * 30)
    print(f"Raw Model mIoU:        {raw_overall*100:.1f}%")
    print(f"Post-processed mIoU:   {processed_overall*100:.1f}%")
    print(f"Overall Improvement:   +{(processed_overall - raw_overall)*100:.1f}%")
    print(f"")
    print(f"Raw Lane mIoU:         {raw_lane*100:.1f}%")
    print(f"Post-processed Lane:   {processed_lane*100:.1f}%")
    print(f"Lane Improvement:      +{(processed_lane - raw_lane)*100:.1f}%")
    
    print(f"\nPER-CLASS COMPARISON:")
    print(f"-" * 40)
    for i, class_name in enumerate(class_names):
        raw_iou = raw_mean[i]
        processed_iou = processed_mean[i]
        improvement = processed_iou - raw_iou
        
        print(f"{class_name:12}: {raw_iou*100:5.1f}% → {processed_iou*100:5.1f}% ({improvement*100:+5.1f}%)")
    
    # Calculate lane coverage statistics
    raw_lane_coverage = []
    processed_lane_coverage = []
    
    for i in range(len(raw_results)):
        # Simulate lane coverage calculation (simplified)
        raw_lane_pixels = raw_results[i][1] + raw_results[i][2]  # white_solid + white_dashed
        processed_lane_pixels = processed_results[i][1] + processed_results[i][2]
        
        raw_lane_coverage.append(raw_lane_pixels)
        processed_lane_coverage.append(processed_lane_pixels)
    
    print(f"\nLANE DETECTION ANALYSIS:")
    print(f"-" * 30)
    print(f"Average Raw Lane Detection:        {np.mean(raw_lane_coverage)*100:.2f}%")
    print(f"Average Processed Lane Detection:  {np.mean(processed_lane_coverage)*100:.2f}%")
    
    # Final assessment
    improvement_threshold = 0.01  # 1% improvement threshold
    
    if processed_overall - raw_overall > improvement_threshold:
        print(f"\n🎉 POST-PROCESSING SUCCESSFUL!")
        print(f"   Significant improvement: +{(processed_overall - raw_overall)*100:.1f}%")
        if processed_overall >= 0.80:
            print(f"   🏆 TARGET ACHIEVED: {processed_overall*100:.1f}% ≥ 80% target!")
        else:
            print(f"   📈 Progress toward 80% target: {processed_overall*100:.1f}%")
    elif abs(processed_overall - raw_overall) <= 0.005:  # Within 0.5%
        print(f"\n✅ POST-PROCESSING NEUTRAL")
        print(f"   Minimal change: {(processed_overall - raw_overall)*100:+.1f}%")
        print(f"   Model already well-optimized at {raw_overall*100:.1f}%")
    else:
        print(f"\n⚠️  POST-PROCESSING NEEDS TUNING")
        print(f"   Decreased performance: {(processed_overall - raw_overall)*100:.1f}%")
        print(f"   Consider adjusting parameters")
    
    print(f"\n" + "=" * 50)
    
    return {
        'raw_miou': raw_overall,
        'processed_miou': processed_overall,
        'improvement': processed_overall - raw_overall,
        'per_class_raw': raw_mean,
        'per_class_processed': processed_mean,
        'samples_tested': sample_count
    }

if __name__ == "__main__":
    results = test_enhanced_post_processing()