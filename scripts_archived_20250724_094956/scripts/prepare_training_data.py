#!/usr/bin/env python3
"""
Phase 3.2: Prepare AEL Dataset for MMSegmentation Training
Creates proper train/val/test splits and converts to 4-class format
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def load_data_splits():
    """Load the train/val/test splits from JSON files."""
    data_dir = Path("data")
    
    with open(data_dir / "train_data.json", 'r') as f:
        train_json = json.load(f)
        train_data = train_json['data']
    
    with open(data_dir / "val_data.json", 'r') as f:
        val_json = json.load(f)
        val_data = val_json['data']
    
    with open(data_dir / "test_data.json", 'r') as f:
        test_json = json.load(f)
        test_data = test_json['data']
    
    print(f"Loaded splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def create_4class_mask(original_mask_path, output_mask_path):
    """Convert original mask to 4-class format for lane detection."""
    # Load original mask
    mask = cv2.imread(str(original_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask {original_mask_path}")
        return False
    
    # Create 4-class mask
    # 0: background, 1: white_solid, 2: white_dashed, 3: yellow_solid
    new_mask = np.zeros_like(mask)
    
    # Map original classes to 4-class system
    # This is a simplified mapping - adjust based on your actual class indices
    class_mapping = {
        0: 0,   # background -> background
        1: 1,   # single_white_solid -> white_solid
        2: 2,   # single_white_dashed -> white_dashed
        3: 3,   # single_yellow_solid -> yellow_solid
        4: 3,   # single_yellow_dashed -> yellow_solid (combine yellow types)
        5: 1,   # double_white_solid -> white_solid (simplify to white)
        6: 3,   # double_yellow_solid -> yellow_solid (simplify to yellow)
        7: 1,   # road_edge -> white_solid
        8: 3,   # center_line -> yellow_solid
        9: 1,   # lane_divider -> white_solid
        10: 1,  # crosswalk -> white_solid
        11: 1   # stop_line -> white_solid
    }
    
    for original_class, new_class in class_mapping.items():
        new_mask[mask == original_class] = new_class
    
    # Save new mask
    cv2.imwrite(str(output_mask_path), new_mask)
    return True

def prepare_split_data(data_list, split_name, source_img_dir, source_mask_dir, target_base_dir):
    """Prepare data for a specific split (train/val/test)."""
    target_img_dir = target_base_dir / "img_dir" / split_name
    target_mask_dir = target_base_dir / "ann_dir" / split_name
    
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_mask_dir.mkdir(parents=True, exist_ok=True)
    
    successful_samples = 0
    
    print(f"Processing {split_name} split...")
    for item in tqdm(data_list, desc=f"Processing {split_name}"):
        # Extract image ID from file path
        # item is a list: [img_path, json_path, mask_path]
        if isinstance(item, list) and len(item) >= 3:
            img_path = item[0]
            mask_path = item[2]
            
            # Extract filename from path
            img_filename = os.path.basename(img_path)
            mask_filename = os.path.basename(mask_path).replace('.jpg', '.png')
            
            # Extract image ID for source lookup
            image_id = os.path.splitext(img_filename)[0]
        else:
            continue
        
        source_img_path = source_img_dir / img_filename
        source_mask_path = source_mask_dir / mask_filename
        
        target_img_path = target_img_dir / img_filename
        target_mask_path = target_mask_dir / mask_filename
        
        # Copy image if exists
        if source_img_path.exists():
            shutil.copy2(source_img_path, target_img_path)
            
            # Create 4-class mask if original mask exists
            if source_mask_path.exists():
                if create_4class_mask(source_mask_path, target_mask_path):
                    successful_samples += 1
            else:
                print(f"Warning: Mask not found for {img_filename}")
        else:
            print(f"Warning: Image not found for {img_filename}")
    
    print(f"Successfully processed {successful_samples} samples for {split_name}")
    return successful_samples

def verify_data_integrity(target_base_dir):
    """Verify that all splits have matching images and masks."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = target_base_dir / "img_dir" / split
        mask_dir = target_base_dir / "ann_dir" / split
        
        img_files = set([f.stem for f in img_dir.glob("*.jpg")])
        mask_files = set([f.stem for f in mask_dir.glob("*.png")])
        
        print(f"\n{split.upper()} Split Verification:")
        print(f"  Images: {len(img_files)}")
        print(f"  Masks: {len(mask_files)}")
        print(f"  Matching pairs: {len(img_files & mask_files)}")
        
        # Check for mismatches
        missing_masks = img_files - mask_files
        missing_images = mask_files - img_files
        
        if missing_masks:
            print(f"  Warning: {len(missing_masks)} images without masks")
        if missing_images:
            print(f"  Warning: {len(missing_images)} masks without images")

def analyze_class_distribution(target_base_dir):
    """Analyze class distribution in the 4-class dataset."""
    splits = ['train', 'val', 'test']
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    
    for split in splits:
        mask_dir = target_base_dir / "ann_dir" / split
        mask_files = list(mask_dir.glob("*.png"))
        
        total_pixels = 0
        class_counts = np.zeros(4)
        
        print(f"\nAnalyzing {split} class distribution...")
        for mask_file in tqdm(mask_files[:100], desc=f"Sampling {split}"):  # Sample first 100
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique, counts = np.unique(mask, return_counts=True)
                total_pixels += mask.size
                
                for class_id, count in zip(unique, counts):
                    if class_id < 4:
                        class_counts[class_id] += count
        
        print(f"{split.upper()} Class Distribution (sampled):")
        for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"  {i}: {class_name}: {count:,} pixels ({percentage:.2f}%)")

def main():
    """Main function to prepare training data."""
    print("Phase 3.2: Preparing AEL Dataset for 4-Class Lane Detection Training")
    
    # Define paths
    source_img_dir = Path("data/imgs")
    source_mask_dir = Path("data/mask")  # Assuming masks are in data/mask/
    target_base_dir = Path("data/ael_mmseg")
    
    # Check if source directories exist
    if not source_img_dir.exists():
        print(f"Error: Source image directory {source_img_dir} does not exist")
        return
    
    # For now, we'll use existing training masks or create dummy ones
    if not source_mask_dir.exists():
        print(f"Warning: Source mask directory {source_mask_dir} does not exist")
        print("Will use existing training masks from data/ael_mmseg/ann_dir/train/")
        source_mask_dir = Path("data/ael_mmseg/ann_dir/train")
    
    # Load data splits
    train_data, val_data, test_data = load_data_splits()
    
    # Prepare each split
    print("\nPreparing Training Data Splits...")
    
    train_count = prepare_split_data(
        train_data[:1000],  # Use subset for initial testing
        "train", 
        source_img_dir, 
        source_mask_dir, 
        target_base_dir
    )
    
    val_count = prepare_split_data(
        val_data[:200],  # Use subset for initial testing
        "val", 
        source_img_dir, 
        source_mask_dir, 
        target_base_dir
    )
    
    test_count = prepare_split_data(
        test_data[:200],  # Use subset for initial testing
        "test", 
        source_img_dir, 
        source_mask_dir, 
        target_base_dir
    )
    
    print(f"\nData Preparation Complete!")
    print(f"   Training samples: {train_count}")
    print(f"   Validation samples: {val_count}")
    print(f"   Test samples: {test_count}")
    
    # Verify data integrity
    verify_data_integrity(target_base_dir)
    
    # Analyze class distribution
    analyze_class_distribution(target_base_dir)
    
    print(f"\nReady for Phase 3.2 Training!")
    print(f"   Config: configs/ael_swin_upernet_training.py")
    print(f"   Target: 80-85% mIoU on 4-class lane detection")

if __name__ == "__main__":
    main()