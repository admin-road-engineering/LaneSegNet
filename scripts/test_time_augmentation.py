#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Enhancement for 85.1% mIoU Model
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

class TTAPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        # TTA transformations
        self.tta_transforms = [
            # Original
            A.Compose([A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            # Horizontal flip
            A.Compose([A.Resize(512, 512), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            # Scale variation
            A.Compose([A.Resize(480, 480), A.Resize(512, 512), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        ]
        
        print(f"TTA Predictor initialized with {len(self.tta_transforms)} augmentations")
    
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
    
    def predict_tta(self, image: np.ndarray):
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
        
        return final_prediction, avg_probabilities

def test_tta_enhancement():
    print("=" * 60)
    print("TEST-TIME AUGMENTATION ENHANCEMENT")
    print("85.1% mIoU Model with TTA Boost")
    print("=" * 60)
    
    model_path = "work_dirs/premium_gpu_best_model.pth"
    if not Path(model_path).exists():
        print("ERROR: Model checkpoint not found!")
        return
    
    predictor = TTAPredictor(model_path)
    
    # Test on Australian images
    downloads_path = Path("C:/Users/Admin/Downloads")
    if downloads_path.exists():
        test_images = list(downloads_path.glob("Australia_*.png"))[:2]
    else:
        test_images = []
    
    if not test_images:
        print("No Australian test images found - using sample from dataset")
        # Try dataset images
        sample_dir = Path("data/ael_mmseg/img_dir/val")
        if sample_dir.exists():
            test_images = list(sample_dir.glob("*.jpg"))[:1]
    
    if not test_images:
        print("No test images available")
        return
    
    for image_file in test_images:
        print(f"\\nTesting TTA on: {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get TTA prediction
        tta_pred, tta_probs = predictor.predict_tta(image)
        
        # Calculate metrics
        white_solid_coverage = (tta_pred == 1).sum() / tta_pred.size * 100
        white_dashed_coverage = (tta_pred == 2).sum() / tta_pred.size * 100
        total_lane_coverage = white_solid_coverage + white_dashed_coverage
        confidence = np.max(tta_probs, axis=0).mean()
        
        print(f"  TTA Results:")
        print(f"    Lane coverage: {total_lane_coverage:.1f}%")
        print(f"    Confidence: {confidence:.3f}")
    
    print("\\n" + "=" * 60)
    print("TTA ENHANCEMENT COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_tta_enhancement()