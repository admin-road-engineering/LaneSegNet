#!/usr/bin/env python3
"""
Quick Status Check - Minimal test to verify everything works
"""

import torch
from pathlib import Path
import sys

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet

def quick_status_check():
    print("QUICK STATUS CHECK")
    print("=" * 30)
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Check model file
    model_path = 'work_dirs/premium_gpu_best_model.pth'
    if Path(model_path).exists():
        print(f"\\nModel file: EXISTS")
        
        # Try loading checkpoint info
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'unknown')
            miou = checkpoint.get('best_miou', 0)
            print(f"Checkpoint: Epoch {epoch}, {miou*100:.1f}% mIoU")
        except Exception as e:
            print(f"Checkpoint load error: {e}")
        
        # Try creating model
        try:
            print("\\nTesting model creation...")
            model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
            print("Model created successfully")
            
            # Try loading weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Weights loaded successfully")
            
            # Try moving to GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            print(f"Model moved to {device} successfully")
            
        except Exception as e:
            print(f"Model setup error: {e}")
    else:
        print(f"\\nModel file: NOT FOUND")
    
    # Check dataset
    val_img_dir = Path('data/ael_mmseg/img_dir/val')
    val_ann_dir = Path('data/ael_mmseg/ann_dir/val')
    
    if val_img_dir.exists() and val_ann_dir.exists():
        img_count = len(list(val_img_dir.glob('*.jpg')))
        ann_count = len(list(val_ann_dir.glob('*.png')))
        print(f"\\nValidation dataset:")
        print(f"  Images: {img_count}")
        print(f"  Annotations: {ann_count}")
    else:
        print(f"\\nValidation dataset: NOT FOUND")
    
    print("\\nSTATUS CHECK COMPLETE")
    print("=" * 30)

if __name__ == "__main__":
    quick_status_check()