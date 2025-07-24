#!/usr/bin/env python3
"""
Quick setup for Phase 3.2 training with existing data
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def setup_training_data():
    """Set up a quick training environment using existing data."""
    print("Setting up training data for Phase 3.2...")
    
    # Use existing training masks from ael_mmseg/ann_dir/train
    source_train_masks = Path("data/ael_mmseg/ann_dir/train")
    source_images = Path("data/imgs")
    
    target_base = Path("data/ael_mmseg")
    
    # Get list of existing masks to determine available samples
    if source_train_masks.exists():
        available_masks = list(source_train_masks.glob("*.png"))
        print(f"Found {len(available_masks)} existing training masks")
        
        # Create splits from available data
        total_masks = len(available_masks)
        train_count = int(total_masks * 0.8)  # 80% for training
        val_count = int(total_masks * 0.1)    # 10% for validation
        # Remaining for test
        
        train_masks = available_masks[:train_count]
        val_masks = available_masks[train_count:train_count + val_count]
        test_masks = available_masks[train_count + val_count:]
        
        print(f"Split: Train={len(train_masks)}, Val={len(val_masks)}, Test={len(test_masks)}")
        
        # Set up validation data
        val_img_dir = target_base / "img_dir" / "val"
        val_mask_dir = target_base / "ann_dir" / "val"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for mask_file in tqdm(val_masks, desc="Setting up validation data"):
            # Copy mask
            shutil.copy2(mask_file, val_mask_dir)
            
            # Copy corresponding image
            img_name = mask_file.name.replace('.png', '.jpg')
            source_img = source_images / img_name
            if source_img.exists():
                shutil.copy2(source_img, val_img_dir)
        
        # Set up test data
        test_img_dir = target_base / "img_dir" / "test"
        test_mask_dir = target_base / "ann_dir" / "test"
        test_img_dir.mkdir(parents=True, exist_ok=True)
        test_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for mask_file in tqdm(test_masks, desc="Setting up test data"):
            # Copy mask
            shutil.copy2(mask_file, test_mask_dir)
            
            # Copy corresponding image
            img_name = mask_file.name.replace('.png', '.jpg')
            source_img = source_images / img_name
            if source_img.exists():
                shutil.copy2(source_img, test_img_dir)
        
        print("✅ Training data setup complete!")
        print(f"   Training: {len(train_masks)} samples (using existing)")
        print(f"   Validation: {len(val_masks)} samples")
        print(f"   Test: {len(test_masks)} samples")
        
        return True
    else:
        print("❌ No existing training masks found")
        return False

def main():
    """Main function."""
    print("Phase 3.2: Quick Training Setup")
    
    if setup_training_data():
        print("\n🚀 Ready to start training!")
        print("Run: python -m mmseg.apis.train configs/ael_swin_upernet_training.py")
    else:
        print("❌ Setup failed - need to create masks first")

if __name__ == "__main__":
    main()