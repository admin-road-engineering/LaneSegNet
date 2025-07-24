#!/usr/bin/env python3
"""
Gentle Post-Processing Pipeline
==============================

A more conservative post-processing approach that preserves model predictions
while making minimal improvements for noise reduction and connectivity.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import cv2
import sys
import time

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

class GentlePostProcessor:
    def __init__(self, min_area=20, kernel_size=3, enable_gap_fill=True):
        """
        Gentle post-processing that preserves existing predictions
        
        Args:
            min_area: Minimum area to remove (very small to preserve details)
            kernel_size: Small kernel for minimal morphological operations
            enable_gap_fill: Whether to enable gap filling
        """
        self.min_area = min_area
        self.kernel_size = kernel_size
        self.enable_gap_fill = enable_gap_fill
        
        # Very small kernels for gentle processing
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Minimal gap filling
        
    def gentle_noise_removal(self, pred_mask):
        """Very gentle noise removal that preserves most predictions"""
        if torch.is_tensor(pred_mask):
            mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            return_tensor = True
        else:
            mask_np = pred_mask.astype(np.uint8)
            return_tensor = False
        
        cleaned = np.copy(mask_np)
        
        # Only process lane classes (preserve background)
        for class_id in [1, 2]:  # white_solid, white_dashed
            class_mask = (mask_np == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Very gentle morphological opening (remove only tiny specks)
            opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, self.small_kernel)
            
            # Remove only very small isolated pixels
            from skimage.morphology import remove_small_objects
            cleaned_class = remove_small_objects(opened.astype(bool), min_size=self.min_area)
            
            # Update cleaned mask
            cleaned[mask_np == class_id] = 0  # Clear original class pixels
            cleaned[cleaned_class] = class_id  # Add cleaned class pixels
        
        return torch.tensor(cleaned) if return_tensor else cleaned
    
    def gentle_gap_filling(self, pred_mask):
        """Minimal gap filling for obvious discontinuities"""
        if not self.enable_gap_fill:
            return pred_mask
            
        if torch.is_tensor(pred_mask):
            mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            return_tensor = True
        else:
            mask_np = pred_mask.astype(np.uint8)
            return_tensor = False
        
        filled = np.copy(mask_np)
        
        # Only fill gaps in lane classes
        for class_id in [1, 2]:
            class_mask = (mask_np == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Very gentle closing to fill only small gaps
            gap_filled = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, self.gap_kernel)
            
            # Only add pixels, don't remove existing ones
            new_pixels = np.logical_and(gap_filled > 0, class_mask == 0)
            filled[new_pixels] = class_id
        
        return torch.tensor(filled) if return_tensor else filled
    
    def process(self, pred_mask):
        """Apply gentle post-processing pipeline"""
        # Step 1: Gentle noise removal
        cleaned = self.gentle_noise_removal(pred_mask)
        
        # Step 2: Minimal gap filling
        final = self.gentle_gap_filling(cleaned)
        
        return final

def calculate_iou_simple(pred_mask, gt_mask):
    """Simple IoU calculation"""
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

def test_gentle_post_processing():
    """Test gentle post-processing pipeline"""
    print("GENTLE POST-PROCESSING EVALUATION")
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
    
    # Test different configurations
    configs = [
        ("No Post-Processing", None),
        ("Gentle (min_area=10)", GentlePostProcessor(min_area=10, kernel_size=3)),
        ("Conservative (min_area=5)", GentlePostProcessor(min_area=5, kernel_size=3)),
        ("Minimal (no gap fill)", GentlePostProcessor(min_area=5, kernel_size=3, enable_gap_fill=False)),
    ]
    
    # Load validation data
    val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    test_samples = 50  # Smaller sample for faster testing
    results = {}
    
    for config_name, post_processor in configs:
        print(f"\nTesting: {config_name}")
        print("-" * 30)
        
        all_ious = []
        sample_count = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                if sample_count >= test_samples:
                    break
                
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for i in range(images.size(0)):
                    if sample_count >= test_samples:
                        break
                    
                    pred = predictions[i]
                    gt = masks[i]
                    
                    # Apply post-processing if specified
                    if post_processor is not None:
                        pred = post_processor.process(pred)
                    
                    # Calculate IoU
                    ious, class_names = calculate_iou_simple(pred, gt)
                    all_ious.append(ious)
                    sample_count += 1
        
        # Calculate statistics
        all_ious = np.array(all_ious)
        mean_ious = np.mean(all_ious, axis=0)
        overall_miou = np.mean(mean_ious)
        lane_miou = np.mean(mean_ious[1:])
        
        results[config_name] = {
            'overall_miou': overall_miou,
            'lane_miou': lane_miou,
            'class_ious': mean_ious
        }
        
        print(f"  Overall mIoU: {overall_miou*100:.1f}%")
        print(f"  Lane mIoU: {lane_miou*100:.1f}%")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {mean_ious[i]*100:.1f}%")
    
    # Comparison summary
    print(f"\nCOMPARISON SUMMARY:")
    print("=" * 50)
    
    baseline = results["No Post-Processing"]
    
    print(f"{'Configuration':<25} {'Overall':<8} {'Lane':<8} {'Change':<8}")
    print("-" * 50)
    
    for config_name, result in results.items():
        overall = result['overall_miou'] * 100
        lane = result['lane_miou'] * 100
        
        if config_name == "No Post-Processing":
            change = "baseline"
        else:
            change = f"{(result['overall_miou'] - baseline['overall_miou'])*100:+.1f}%"
        
        print(f"{config_name:<25} {overall:>6.1f}%  {lane:>6.1f}%  {change:>7}")
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['overall_miou'])
    best_name, best_result = best_config
    
    print(f"\nBEST CONFIGURATION: {best_name}")
    print(f"Performance: {best_result['overall_miou']*100:.1f}% mIoU")
    
    if best_result['overall_miou'] > baseline['overall_miou']:
        improvement = (best_result['overall_miou'] - baseline['overall_miou']) * 100
        print(f"Improvement: +{improvement:.1f}%")
        
        if best_result['overall_miou'] >= 0.80:
            print("TARGET ACHIEVED: >= 80% mIoU!")
        else:
            print(f"Progress toward 80% target: {best_result['overall_miou']*100:.1f}%")
    else:
        print("No improvement found - raw model is already optimal")
    
    return results

if __name__ == "__main__":
    results = test_gentle_post_processing()