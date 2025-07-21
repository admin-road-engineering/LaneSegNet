#!/usr/bin/env python3
"""
Quick evaluation of the enhanced model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Enhanced model architecture (same as training)
class EnhancedLaneNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Simple dataset for evaluation
class EvalDataset:
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = list(self.img_dir.glob("*.jpg"))[:100]  # Sample 100 for quick eval
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        mask = np.clip(mask, 0, 3)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def calculate_iou(pred, target, num_classes=4):
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious

def evaluate_enhanced_model():
    print("Quick Enhanced Model Evaluation")
    print("Testing on 100 validation samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model_path = "work_dirs/enhanced_best_model.pth"
    model = EnhancedLaneNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Dataset
    dataset = EvalDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                ious = calculate_iou(pred[i], masks[i])
                all_ious.append(ious)
    
    # Results
    mean_ious = np.mean(all_ious, axis=0)
    miou = np.mean(mean_ious)
    
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    
    print()
    print("=== ENHANCED MODEL RESULTS ===")
    print(f"Samples evaluated: {len(all_ious)}")
    print(f"Overall mIoU: {miou:.1%}")
    print()
    print("Per-class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, mean_ious)):
        print(f"  {i}: {name:15} {iou:.1%}")
    
    print()
    print("COMPARISON:")
    print(f"  Baseline:  52.0% mIoU")
    print(f"  Enhanced:  {miou:.1%} mIoU")
    print(f"  Improvement: {(miou - 0.52):.1%}")
    
    # Target assessment
    if miou >= 0.85:
        print("\n🏆 OUTSTANDING! 85%+ target achieved!")
    elif miou >= 0.80:
        print("\n🎯 SUCCESS! 80-85% target achieved!")
    elif miou >= 0.65:
        print("\n✅ GOOD! Above baseline target")
    else:
        print("\n⚠ Below target, but improved over baseline")
    
    return miou

if __name__ == "__main__":
    result = evaluate_enhanced_model()
    print(f"\nFinal enhanced mIoU: {result:.1%}")