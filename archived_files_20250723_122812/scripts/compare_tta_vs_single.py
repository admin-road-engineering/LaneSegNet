#!/usr/bin/env python3
"""
Compare TTA vs Single Prediction against Ground Truth
Test on 100 random images from data/imgs with ground truth from data/json
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys
import random
from typing import Dict, List
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet

class ModelComparer:
    """Compare single prediction vs TTA against ground truth"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        # Standard preprocessing
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # TTA transformations
        self.tta_transforms = [
            # Original
            A.Compose([A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            # Horizontal flip
            A.Compose([A.Resize(512, 512), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            # Scale variation
            A.Compose([A.Resize(480, 480), A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        ]
        
        print(f"Model loaded for comparison testing")
    
    def _load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu')
        model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        miou = checkpoint.get('best_miou', 0)
        print(f"Loaded model: Epoch {epoch}, mIoU: {miou*100:.1f}%")
        return model
    
    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Single prediction without TTA"""
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return prediction[0].cpu().numpy()
    
    def predict_tta(self, image: np.ndarray) -> np.ndarray:
        """TTA prediction with multiple augmentations"""
        all_probabilities = []
        
        for i, transform in enumerate(self.tta_transforms):
            transformed = transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Handle inverse transformations
            if i == 1:  # Horizontal flip - flip back
                probabilities = torch.flip(probabilities, dims=[3])
            
            all_probabilities.append(probabilities[0].cpu().numpy())
        
        # Average all probability maps
        avg_probabilities = np.mean(all_probabilities, axis=0)
        final_prediction = np.argmax(avg_probabilities, axis=0)
        
        return final_prediction
    
    def load_ground_truth(self, mask_path: Path) -> np.ndarray:
        """Load ground truth from mask file (same as training)"""
        try:
            # Load the mask file directly
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            
            # Resize to match prediction size
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            return mask
            
        except Exception as e:
            print(f"Error loading ground truth {mask_path}: {e}")
            return None
    
    def calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """Calculate IoU for each class"""
        ious = {}
        class_names = ['background', 'white_solid', 'white_dashed']
        
        for class_id, class_name in enumerate(class_names):
            pred_class = (pred == class_id)
            gt_class = (gt == class_id)
            
            intersection = np.logical_and(pred_class, gt_class).sum()
            union = np.logical_or(pred_class, gt_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious[class_name] = iou
        
        # Overall mIoU
        ious['mean_iou'] = np.mean(list(ious.values())[:3])  # Exclude mean from mean calculation
        
        return ious

def run_comparison_test():
    """Run comprehensive comparison test"""
    print("=" * 80)
    print("COMPREHENSIVE TTA vs SINGLE PREDICTION COMPARISON")
    print("Testing on 100 random images with ground truth")
    print("=" * 80)
    
    # Initialize comparer
    model_path = "work_dirs/premium_gpu_best_model.pth"
    if not Path(model_path).exists():
        print("ERROR: Model checkpoint not found!")
        return
    
    comparer = ModelComparer(model_path)
    
    # Get image and mask paths (using training format)
    images_dir = Path("data/ael_mmseg/img_dir/val")  # Use validation set
    masks_dir = Path("data/ael_mmseg/ann_dir/val")   # Ground truth masks
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"ERROR: Required directories not found:")
        print(f"  Images: {images_dir} ({'exists' if images_dir.exists() else 'missing'})")
        print(f"  Masks: {masks_dir} ({'exists' if masks_dir.exists() else 'missing'})")
        return
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} total images")
    
    if len(image_files) < 100:
        print(f"Warning: Only {len(image_files)} images available, using all")
        test_images = image_files
    else:
        # Randomly sample 100 images
        random.seed(42)  # For reproducible results
        test_images = random.sample(image_files, 100)
    
    print(f"Testing on {len(test_images)} images...")
    print()
    
    # Results storage
    single_results = []
    tta_results = []
    valid_tests = 0
    
    # Test each image
    for i, image_path in enumerate(test_images):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(test_images)} images processed...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        
        gt_mask = comparer.load_ground_truth(mask_path)
        if gt_mask is None:
            continue
        
        # Get predictions
        single_pred = comparer.predict_single(image)
        tta_pred = comparer.predict_tta(image)
        
        # Calculate IoUs
        single_iou = comparer.calculate_iou(single_pred, gt_mask)
        tta_iou = comparer.calculate_iou(tta_pred, gt_mask)
        
        single_results.append(single_iou)
        tta_results.append(tta_iou)
        valid_tests += 1
    
    print(f"Completed: {valid_tests} valid tests")
    print()
    
    if valid_tests == 0:
        print("ERROR: No valid test cases found!")
        return
    
    # Calculate average results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    class_names = ['background', 'white_solid', 'white_dashed', 'mean_iou']
    
    print(f"{'Method':<15} {'Background':<12} {'White Solid':<12} {'White Dashed':<13} {'Mean IoU':<10}")
    print("-" * 65)
    
    # Single prediction averages
    single_avgs = {}
    for class_name in class_names:
        single_avgs[class_name] = np.mean([r[class_name] for r in single_results])
    
    print(f"{'Single':<15} {single_avgs['background']:<12.3f} {single_avgs['white_solid']:<12.3f} {single_avgs['white_dashed']:<13.3f} {single_avgs['mean_iou']:<10.3f}")
    
    # TTA averages
    tta_avgs = {}
    for class_name in class_names:
        tta_avgs[class_name] = np.mean([r[class_name] for r in tta_results])
    
    print(f"{'TTA':<15} {tta_avgs['background']:<12.3f} {tta_avgs['white_solid']:<12.3f} {tta_avgs['white_dashed']:<13.3f} {tta_avgs['mean_iou']:<10.3f}")
    
    # Improvements
    print()
    print("IMPROVEMENTS (TTA vs Single):")
    for class_name in class_names:
        improvement = tta_avgs[class_name] - single_avgs[class_name]
        print(f"  {class_name}: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    # Summary
    print()
    print("SUMMARY:")
    if tta_avgs['mean_iou'] > single_avgs['mean_iou']:
        improvement = (tta_avgs['mean_iou'] - single_avgs['mean_iou']) * 100
        print(f"SUCCESS: TTA improves performance by {improvement:.1f}% mIoU")
    else:
        decline = (single_avgs['mean_iou'] - tta_avgs['mean_iou']) * 100
        print(f"WARNING: TTA reduces performance by {decline:.1f}% mIoU")
    
    print(f"  Single prediction mIoU: {single_avgs['mean_iou']*100:.1f}%")
    print(f"  TTA prediction mIoU: {tta_avgs['mean_iou']*100:.1f}%")
    print(f"  Tested on {valid_tests} images")
    
    print("=" * 80)

if __name__ == "__main__":
    run_comparison_test()