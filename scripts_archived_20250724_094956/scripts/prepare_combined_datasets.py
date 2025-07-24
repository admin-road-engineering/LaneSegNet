#!/usr/bin/env python3
"""
Combined Dataset Preparation - Merge AEL, SS_Dense, and SS_Multi_Lane for training
Converts all datasets to unified format for lane detection training
"""

import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from PIL import Image

def create_combined_directory_structure():
    """Create unified dataset directory structure"""
    base_dir = Path("data/combined_lane_dataset")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    print(f"Created combined dataset directory: {base_dir}")
    return base_dir

def convert_skyscapes_mask_to_lane_classes(rgb_mask_path):
    """
    Convert SkyScapes RGB mask to our 3-class lane system:
    Class 0: Background (everything else)
    Class 1: White solid lines (Long Line - class 2 in SkyScapes-Lane)
    Class 2: White dashed lines (Dash Line - class 1 in SkyScapes-Lane)
    """
    # Load RGB mask
    rgb_mask = cv2.imread(str(rgb_mask_path))
    if rgb_mask is None:
        return None
    
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    h, w = rgb_mask.shape[:2]
    
    # Create output mask (grayscale with class indices)
    output_mask = np.zeros((h, w), dtype=np.uint8)
    
    # SkyScapes-Lane color mappings
    # Class 1: Dash Line (red) -> Our Class 2 (white dashed)
    dash_line_mask = np.all(rgb_mask == [255, 0, 0], axis=2)
    output_mask[dash_line_mask] = 2
    
    # Class 2: Long Line (blue) -> Our Class 1 (white solid)  
    long_line_mask = np.all(rgb_mask == [0, 0, 255], axis=2)
    output_mask[long_line_mask] = 1
    
    # Small dash lines (yellow) -> Also treat as dashed
    small_dash_mask = np.all(rgb_mask == [255, 255, 0], axis=2)
    output_mask[small_dash_mask] = 2
    
    # All other lane markings -> treat as solid lines
    other_markings_colors = [
        [0, 255, 0],    # Turn signs
        [255, 128, 0],  # Other signs
        [128, 0, 0],    # Plus sign
        [0, 255, 255],  # Crosswalk  
        [0, 128, 0],    # Stop line
        [255, 0, 255],  # Zebra zone
        [0, 150, 150],  # No parking
        [200, 200, 0],  # Parking space
        [100, 0, 200]   # Other lane markings
    ]
    
    for color in other_markings_colors:
        mask = np.all(rgb_mask == color, axis=2)
        output_mask[mask] = 1  # Treat as solid lines
    
    return output_mask

def process_skyscapes_dataset(source_dir, target_dir, dataset_name, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
    """Process SkyScapes dataset (SS_Dense or SS_Multi_Lane)"""
    print(f"\n=== Processing {dataset_name} Dataset ===")
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Get all training images (we'll create our own splits)
    image_dir = source_dir / "train" / "images"
    label_dir = source_dir / "train" / "labels" / "rgb"
    
    if not image_dir.exists() or not label_dir.exists():
        print(f"ERROR: Missing directories in {source_dir}")
        return 0
    
    # Get all image files
    image_files = sorted(list(image_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} training images")
    
    # Add validation and test images if they exist
    for split_name in ['val', 'test']:
        split_image_dir = source_dir / split_name / "images"
        if split_image_dir.exists():
            split_images = sorted(list(split_image_dir.glob("*.jpg")))
            image_files.extend(split_images)
            print(f"Added {len(split_images)} {split_name} images")
    
    if len(image_files) == 0:
        print(f"No images found in {source_dir}")
        return 0
    
    # Create splits
    n_total = len(image_files)
    n_train = int(n_total * split_ratios['train'])
    n_val = int(n_total * split_ratios['val'])
    n_test = n_total - n_train - n_val
    
    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Process each split
    processed_count = 0
    
    for i, image_file in enumerate(tqdm(image_files, desc=f"Processing {dataset_name}")):
        # Determine split
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
        
        # Find corresponding label file
        label_file = None
        for possible_label_dir in [source_dir / "train" / "labels" / "rgb",
                                  source_dir / "val" / "labels" / "rgb", 
                                  source_dir / "test" / "labels" / "rgb"]:
            potential_label = possible_label_dir / f"{image_file.stem}.png"
            if potential_label.exists():
                label_file = potential_label
                break
        
        if label_file is None:
            print(f"Warning: No label found for {image_file.name}")
            continue
        
        # Convert mask to our format
        converted_mask = convert_skyscapes_mask_to_lane_classes(label_file)
        if converted_mask is None:
            print(f"Warning: Could not convert mask for {image_file.name}")
            continue
        
        # Create output filenames
        output_name = f"{dataset_name}_{image_file.stem}"
        target_image = target_dir / split / 'images' / f"{output_name}.jpg"
        target_mask = target_dir / split / 'masks' / f"{output_name}.png"
        
        # Copy image
        shutil.copy2(image_file, target_image)
        
        # Save converted mask
        cv2.imwrite(str(target_mask), converted_mask)
        
        processed_count += 1
    
    print(f"Successfully processed {processed_count} samples from {dataset_name}")
    return processed_count

def process_ael_dataset(source_dir, target_dir, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
    """Process existing AEL dataset"""
    print(f"\n=== Processing AEL Dataset ===")
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Use existing AEL mmseg format
    ael_img_dir = source_dir / "ael_mmseg" / "img_dir" / "train"
    ael_ann_dir = source_dir / "ael_mmseg" / "ann_dir" / "train"
    
    if not ael_img_dir.exists() or not ael_ann_dir.exists():
        print(f"ERROR: AEL dataset not found in {source_dir}")
        return 0
    
    # Get all images
    image_files = sorted(list(ael_img_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} AEL images")
    
    # Create splits
    n_total = len(image_files)
    n_train = int(n_total * split_ratios['train'])
    n_val = int(n_total * split_ratios['val'])
    n_test = n_total - n_train - n_val
    
    print(f"AEL split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    processed_count = 0
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing AEL")):
        # Determine split
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
        
        # Find corresponding mask
        mask_file = ael_ann_dir / f"{image_file.stem}.png"
        if not mask_file.exists():
            print(f"Warning: No mask found for {image_file.name}")
            continue
        
        # Create output filenames
        output_name = f"AEL_{image_file.stem}"
        target_image = target_dir / split / 'images' / f"{output_name}.jpg"
        target_mask = target_dir / split / 'masks' / f"{output_name}.png"
        
        # Copy files
        shutil.copy2(image_file, target_image)
        shutil.copy2(mask_file, target_mask)
        
        processed_count += 1
    
    print(f"Successfully processed {processed_count} samples from AEL")
    return processed_count

def create_dataset_info(base_dir, counts):
    """Create dataset information file"""
    info = {
        "dataset_name": "Combined Lane Detection Dataset",
        "version": "1.0",
        "description": "Combined dataset from AEL, SS_Dense, and SS_Multi_Lane for improved lane detection",
        "classes": {
            0: "background",
            1: "white_solid_lane", 
            2: "white_dashed_lane"
        },
        "datasets_included": list(counts.keys()),
        "total_samples": sum(counts.values()),
        "samples_per_dataset": counts,
        "splits": {
            "train": "70%",
            "val": "15%", 
            "test": "15%"
        }
    }
    
    # Count actual files
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split / 'images'
        if split_dir.exists():
            split_count = len(list(split_dir.glob("*.jpg")))
            info[f"{split}_samples"] = split_count
    
    # Save info
    info_file = base_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDataset info saved to: {info_file}")
    return info

def main():
    print("=" * 80)
    print("COMBINED DATASET PREPARATION")
    print("Merging AEL + SS_Dense + SS_Multi_Lane for enhanced training")
    print("=" * 80)
    
    # Create combined dataset directory
    target_dir = create_combined_directory_structure()
    
    # Process each dataset
    dataset_counts = {}
    
    # Process AEL dataset
    ael_count = process_ael_dataset("data", target_dir)
    dataset_counts["AEL"] = ael_count
    
    # Process SS_Dense dataset  
    dense_count = process_skyscapes_dataset("data/SS_Dense", target_dir, "SS_Dense")
    dataset_counts["SS_Dense"] = dense_count
    
    # Process SS_Multi_Lane dataset
    multi_count = process_skyscapes_dataset("data/SS_Multi_Lane", target_dir, "SS_Multi_Lane")  
    dataset_counts["SS_Multi_Lane"] = multi_count
    
    # Create dataset info
    info = create_dataset_info(target_dir, dataset_counts)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total samples processed: {sum(dataset_counts.values())}")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} samples")
    print(f"\nCombined dataset location: {target_dir}")
    print(f"Ready for training with enhanced multi-dataset approach!")
    
if __name__ == "__main__":
    main()