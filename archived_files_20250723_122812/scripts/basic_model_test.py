#!/usr/bin/env python3
"""
Basic Model Test - Test with a single image file
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet

def basic_test():
    print("BASIC MODEL TEST - SINGLE IMAGE")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    reported_miou = checkpoint.get('best_miou', 0)
    print(f"Model loaded! Reported mIoU: {reported_miou*100:.1f}%")
    
    # Find a single validation image
    val_img_dir = Path('data/ael_mmseg/img_dir/val')
    val_ann_dir = Path('data/ael_mmseg/ann_dir/val')
    
    image_files = list(val_img_dir.glob('*.jpg'))
    if not image_files:
        print("No validation images found!")
        return
    
    # Test first image
    test_image = image_files[0]
    test_mask = val_ann_dir / f"{test_image.stem}.png"
    
    print(f"Testing image: {test_image.name}")
    
    # Load and preprocess image
    print("Loading image...")
    image = cv2.imread(str(test_image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load ground truth
    gt_mask = cv2.imread(str(test_mask), cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing (same as training)
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print("Preprocessing...")
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Resize ground truth to match
    gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    pred_mask = prediction[0].cpu().numpy()
    conf_avg = torch.max(probabilities, dim=1)[0].mean().item()
    
    print(f"Inference completed! Average confidence: {conf_avg:.3f}")
    
    # Calculate IoU for this single image
    print("Calculating IoU...")
    
    class_names = ['background', 'white_solid', 'white_dashed']
    ious = []
    
    for class_id, class_name in enumerate(class_names):
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)
        
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        print(f"  {class_name}: {iou*100:.1f}% IoU")
    
    overall_miou = np.mean(ious)
    
    # Pixel statistics
    pred_lane_pixels = ((pred_mask == 1) | (pred_mask == 2)).sum()
    gt_lane_pixels = ((gt_mask == 1) | (gt_mask == 2)).sum()
    total_pixels = pred_mask.size
    
    print(f"\\nRESULTS FOR SINGLE IMAGE:")
    print(f"  Overall mIoU: {overall_miou*100:.1f}%")
    print(f"  Predicted lane coverage: {pred_lane_pixels/total_pixels*100:.1f}%")
    print(f"  Ground truth lane coverage: {gt_lane_pixels/total_pixels*100:.1f}%")
    print(f"  Average confidence: {conf_avg:.3f}")
    
    print(f"\\nCOMPARISON:")
    print(f"  Reported training mIoU: {reported_miou*100:.1f}%")
    print(f"  Single image mIoU: {overall_miou*100:.1f}%")
    print(f"  Note: Single image results can vary significantly")
    
    print("\\nBASIC TEST COMPLETED!")
    return overall_miou

if __name__ == "__main__":
    basic_test()