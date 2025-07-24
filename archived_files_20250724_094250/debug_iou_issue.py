#!/usr/bin/env python3
"""
Quick debug script to investigate the IoU calculation issue.
"""
import torch
import numpy as np
from pathlib import Path
import cv2
import sys
sys.path.append(str(Path(__file__).parent))

def check_dataset_classes():
    """Check what classes are actually in our dataset."""
    from data.labeled_dataset import LabeledDataset
    
    # Create a small dataset to sample from
    val_dataset = LabeledDataset(
        "data/full_ael_mmseg/img_dir/val",
        "data/full_ael_mmseg/ann_dir/val", 
        mode='val',
        img_size=(512, 512)
    )
    
    print(f"Dataset size: {len(val_dataset)}")
    
    # Sample a few images to check class distribution
    unique_classes = set()
    for i in range(min(10, len(val_dataset))):
        image, mask = val_dataset[i]
        mask_np = mask.numpy()
        classes_in_sample = np.unique(mask_np)
        unique_classes.update(classes_in_sample)
        print(f"Sample {i}: classes found = {classes_in_sample}")
    
    print(f"\nAll unique classes in dataset: {sorted(unique_classes)}")
    return sorted(unique_classes)

def check_model_output():
    """Check what the model is actually outputting."""
    from scripts.run_finetuning import PretrainedLaneNet
    
    model = PretrainedLaneNet(
        num_classes=3,
        img_size=512,
        encoder_weights_path="work_dirs/mae_pretraining/mae_best_model.pth"
    )
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        output = model(dummy_input)
        predictions = torch.argmax(output, dim=1)
        
    print(f"Model output shape: {output.shape}")
    print(f"Model output classes: {output.shape[1]}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Unique predictions in dummy output: {torch.unique(predictions)}")
    
    return output.shape[1]

def test_iou_calculation():
    """Test the IoU calculation with some dummy data."""
    
    def calculate_iou_test(pred, target, num_classes):
        pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        target_np = target.cpu().numpy() if torch.is_tensor(target) else target
        
        print(f"Pred unique: {np.unique(pred_np)}")
        print(f"Target unique: {np.unique(target_np)}")
        
        ious = []
        for class_id in range(1, num_classes):  # Skip background (class 0)
            pred_mask = (pred_np == class_id)
            target_mask = (target_np == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            print(f"Class {class_id}: intersection={intersection}, union={union}")
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                print(f"Class {class_id} IoU: {iou:.4f}")
            else:
                print(f"Class {class_id}: No pixels found")
        
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        print(f"Mean IoU: {mean_iou:.4f}")
        return mean_iou
    
    # Test with dummy data
    print("=== Testing IoU calculation ===")
    dummy_pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2], [2, 2, 0, 0], [1, 1, 2, 2]])
    dummy_target = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [2, 2, 0, 0], [1, 1, 1, 2]])
    
    calculate_iou_test(dummy_pred, dummy_target, 3)

if __name__ == "__main__":
    print("=== Debugging IoU Issue ===\n")
    
    print("1. Checking dataset classes...")
    try:
        dataset_classes = check_dataset_classes()
    except Exception as e:
        print(f"Error checking dataset: {e}")
        dataset_classes = []
    
    print("\n2. Checking model output...")
    try:
        model_classes = check_model_output()
    except Exception as e:
        print(f"Error checking model: {e}")
        model_classes = 3
    
    print("\n3. Testing IoU calculation...")
    test_iou_calculation()
    
    print(f"\n=== Summary ===")
    print(f"Dataset classes: {dataset_classes}")  
    print(f"Model output classes: {model_classes}")
    print(f"Expected: Background=0, Lane classes=1,2,...")