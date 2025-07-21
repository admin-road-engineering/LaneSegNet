#!/usr/bin/env python3
"""
Evaluate baseline performance for Phase 3.2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm

class SimpleAELDataset:
    """Simple dataset for evaluation."""
    
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        # Load and resize
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

class SimpleLaneNet(nn.Module):
    """Simple lane detection network."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

def calculate_iou(pred, target, num_classes=4):
    """Calculate IoU for each class."""
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

def evaluate_model():
    """Evaluate the trained model."""
    print("Phase 3.2: Baseline Evaluation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model exists
    model_path = "work_dirs/best_model.pth"
    if not Path(model_path).exists():
        print("❌ No trained model found. Run training first.")
        return None
    
    # Load model
    model = SimpleLaneNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Validation dataset
    val_dataset = SimpleAELDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Evaluate
    all_ious = []
    
    print("Evaluating on validation set...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                ious = calculate_iou(pred[i], masks[i])
                all_ious.append(ious)
    
    # Calculate final metrics
    mean_ious = np.mean(all_ious, axis=0)
    miou = np.mean(mean_ious)
    
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Overall mIoU: {miou:.1%}")
    print("\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, mean_ious)):
        print(f"  {i}: {name:12} {iou:.1%}")
    
    # Check baseline target
    baseline_target = 0.65
    if miou >= baseline_target:
        print(f"\n✓ BASELINE TARGET ACHIEVED!")
        print(f"  Target: {baseline_target:.1%}")
        print(f"  Achieved: {miou:.1%}")
        status = "SUCCESS"
    else:
        print(f"\n⚠ BELOW BASELINE TARGET")
        print(f"  Target: {baseline_target:.1%}")
        print(f"  Achieved: {miou:.1%}")
        print(f"  Gap: {(baseline_target - miou):.1%}")
        status = "NEEDS_IMPROVEMENT"
    
    # Save results
    results = {
        "miou": miou,
        "class_ious": mean_ious.tolist(),
        "class_names": class_names,
        "baseline_target": baseline_target,
        "status": status,
        "num_val_samples": len(all_ious)
    }
    
    with open("work_dirs/baseline_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return miou

if __name__ == "__main__":
    result = evaluate_model()
    if result is not None:
        print(f"\nNext steps:")
        if result >= 0.65:
            print("🎯 Ready for Phase 3.2 full training with Swin Transformer")
        else:
            print("🔧 Need to improve baseline before scaling up")