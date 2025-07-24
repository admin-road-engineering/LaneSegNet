#!/usr/bin/env python3
"""
Create proper training data for fine-tuning from original premium training setup
"""

import json
import os
from pathlib import Path
import random

def create_training_data():
    """Create training/validation splits from available images"""
    
    # Check available images and masks
    img_dir = Path("data/imgs")
    mask_dir = Path("data/mask")
    
    if not img_dir.exists() or not mask_dir.exists():
        print(f"❌ Missing directories: {img_dir} or {mask_dir}")
        return
    
    # Find all available image-mask pairs
    available_pairs = []
    
    for img_file in img_dir.glob("*.jpg"):
        img_id = img_file.stem
        mask_file = mask_dir / f"{img_id}.jpg"
        
        if mask_file.exists():
            available_pairs.append(img_id)
    
    print(f"Found {len(available_pairs)} image-mask pairs")
    
    if len(available_pairs) < 100:
        print("❌ Insufficient data for training")
        return
    
    # Shuffle and split
    random.seed(42)  # Reproducible splits
    random.shuffle(available_pairs)
    
    # 70/15/15 split for fine-tuning (smaller validation set)
    total = len(available_pairs)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    
    train_ids = available_pairs[:train_size]
    val_ids = available_pairs[train_size:train_size + val_size]
    test_ids = available_pairs[train_size + val_size:]
    
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Create data files
    def create_data_file(ids, filename):
        data = []
        for img_id in ids:
            data.append({
                "id": img_id,
                "image_path": f"data/imgs/{img_id}.jpg",
                "mask_path": f"data/mask/{img_id}.jpg"
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Created {filename}: {len(data)} samples")
    
    create_data_file(train_ids, "data/train_data.json")
    create_data_file(val_ids, "data/val_data.json") 
    create_data_file(test_ids, "data/test_data.json")
    
    print(f"✅ Fine-tuning data created successfully!")

if __name__ == "__main__":
    create_training_data()