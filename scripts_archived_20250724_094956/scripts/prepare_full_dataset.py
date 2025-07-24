#!/usr/bin/env python3
"""
Full Dataset Preparation
=======================

Prepare the complete 7,819 sample dataset with proper 70/10/20 split
for training, validation, and testing to maximize model performance.
"""

import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
import cv2

def analyze_full_dataset():
    """Analyze the full dataset available"""
    print("FULL DATASET ANALYSIS")
    print("=" * 40)
    
    img_dir = Path('data/imgs')
    json_dir = Path('data/json')
    
    # Count files
    img_files = list(img_dir.glob('*.jpg'))
    json_files = list(json_dir.glob('*.json'))
    
    print(f"Available images: {len(img_files)}")
    print(f"Available JSON annotations: {len(json_files)}")
    
    # Find matched pairs
    img_ids = {f.stem for f in img_files}
    json_ids = {f.stem for f in json_files}
    
    matched_ids = img_ids.intersection(json_ids)
    missing_json = img_ids - json_ids
    missing_img = json_ids - img_ids
    
    print(f"Matched pairs: {len(matched_ids)}")
    if missing_json:
        print(f"Images without JSON: {len(missing_json)}")
    if missing_img:
        print(f"JSON without images: {len(missing_img)}")
    
    # Analyze a few samples for quality
    sample_ids = list(matched_ids)[:10]
    valid_samples = 0
    
    print(f"\nQuality check on {len(sample_ids)} samples...")
    for sample_id in sample_ids:
        try:
            # Check image
            img_path = img_dir / f"{sample_id}.jpg"
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue
                
            # Check JSON
            json_path = json_dir / f"{sample_id}.json"
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if 'shapes' in data and len(data['shapes']) > 0:
                valid_samples += 1
                
        except Exception as e:
            print(f"  Error with {sample_id}: {e}")
    
    print(f"Valid samples: {valid_samples}/{len(sample_ids)}")
    
    return {
        'total_matched': len(matched_ids),
        'matched_ids': list(matched_ids),
        'quality_ratio': valid_samples / len(sample_ids) if sample_ids else 0
    }

def create_segmentation_masks():
    """Create segmentation masks from JSON annotations"""
    print(f"\nCreating segmentation masks...")
    
    img_dir = Path('data/imgs')
    json_dir = Path('data/json')
    mask_dir = Path('data/full_masks')
    mask_dir.mkdir(exist_ok=True)
    
    dataset_info = analyze_full_dataset()
    matched_ids = dataset_info['matched_ids']
    
    print(f"Creating masks for {len(matched_ids)} samples...")
    
    created_masks = 0
    failed_masks = 0
    
    for i, sample_id in enumerate(matched_ids):
        try:
            # Load image to get dimensions
            img_path = img_dir / f"{sample_id}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                failed_masks += 1
                continue
                
            h, w = img.shape[:2]
            
            # Load JSON annotation
            json_path = json_dir / f"{sample_id}.json"
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Process each shape in the JSON
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                points = np.array(shape.get('points', []), dtype=np.int32)
                
                if len(points) < 3:  # Need at least 3 points for polygon
                    continue
                
                # Map labels to class IDs
                class_id = 0  # background
                if 'white' in label and 'solid' in label:
                    class_id = 1
                elif 'white' in label and ('dash' in label or 'dot' in label):
                    class_id = 2
                elif 'yellow' in label:
                    class_id = 1  # Treat yellow as white solid for simplicity
                
                if class_id > 0:
                    cv2.fillPoly(mask, [points], class_id)
            
            # Save mask
            mask_path = mask_dir / f"{sample_id}.png"
            cv2.imwrite(str(mask_path), mask)
            created_masks += 1
            
            if (i + 1) % 500 == 0:
                print(f"  Created {i + 1}/{len(matched_ids)} masks...")
                
        except Exception as e:
            failed_masks += 1
            if failed_masks < 10:  # Only print first 10 errors
                print(f"  Error creating mask for {sample_id}: {e}")
    
    print(f"Mask creation complete:")
    print(f"  Created: {created_masks}")
    print(f"  Failed: {failed_masks}")
    
    return created_masks

def create_train_val_test_split(matched_ids, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20):
    """Create proper train/val/test split"""
    
    print(f"\nCreating dataset split...")
    print(f"  Total samples: {len(matched_ids)}")
    print(f"  Target split: {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}% (train/val/test)")
    
    # Shuffle the matched IDs for random split
    random.seed(42)
    shuffled_ids = matched_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split indices
    total = len(shuffled_ids)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    # Create splits
    train_ids = shuffled_ids[:train_end]
    val_ids = shuffled_ids[train_end:val_end]
    test_ids = shuffled_ids[val_end:]
    
    print(f"Actual split:")
    print(f"  Training: {len(train_ids)} samples ({len(train_ids)/len(matched_ids)*100:.1f}%)")
    print(f"  Validation: {len(val_ids)} samples ({len(val_ids)/len(matched_ids)*100:.1f}%)")
    print(f"  Test: {len(test_ids)} samples ({len(test_ids)/len(matched_ids)*100:.1f}%)")
    
    return train_ids, val_ids, test_ids

def create_mmseg_dataset(train_ids, val_ids, test_ids):
    """Create MMSegmentation format dataset"""
    
    print(f"\nCreating MMSegmentation dataset structure...")
    
    # Create directory structure
    base_dir = Path('data/full_ael_mmseg')
    
    dirs_to_create = [
        'img_dir/train',
        'img_dir/val', 
        'img_dir/test',
        'ann_dir/train',
        'ann_dir/val',
        'ann_dir/test'
    ]
    
    for dir_name in dirs_to_create:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate directories
    img_dir = Path('data/imgs')
    mask_dir = Path('data/full_masks')
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, ids in splits.items():
        print(f"  Creating {split_name} split ({len(ids)} samples)...")
        
        img_dest = base_dir / 'img_dir' / split_name
        ann_dest = base_dir / 'ann_dir' / split_name
        
        for sample_id in ids:
            # Copy image
            src_img = img_dir / f"{sample_id}.jpg"
            dst_img = img_dest / f"{sample_id}.jpg"
            
            # Copy mask
            src_mask = mask_dir / f"{sample_id}.png"
            dst_mask = ann_dest / f"{sample_id}.png"
            
            try:
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_mask, dst_mask)
            except Exception as e:
                print(f"    Error copying {sample_id}: {e}")
    
    print(f"MMSegmentation dataset created at: {base_dir}")
    return base_dir

def generate_dataset_report(base_dir, train_ids, val_ids, test_ids):
    """Generate comprehensive dataset report"""
    
    report = {
        'dataset_info': {
            'total_samples': len(train_ids) + len(val_ids) + len(test_ids),
            'train_samples': len(train_ids),
            'val_samples': len(val_ids), 
            'test_samples': len(test_ids),
            'split_ratios': {
                'train': len(train_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
                'val': len(val_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
                'test': len(test_ids) / (len(train_ids) + len(val_ids) + len(test_ids))
            }
        },
        'directory_structure': {
            'base_path': str(base_dir),
            'img_dir': str(base_dir / 'img_dir'),
            'ann_dir': str(base_dir / 'ann_dir')
        },
        'comparison_with_current': {
            'current_train': 5471,
            'current_val': 1328,
            'new_train': len(train_ids),
            'new_val': len(val_ids),
            'improvement_train': f"{(len(train_ids)/5471 - 1)*100:.0f}%",
            'improvement_val': f"{(len(val_ids)/1328 - 1)*100:.0f}%"
        }
    }
    
    # Save report
    report_path = f"full_dataset_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*50)
    print("FULL DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"Dataset location: {base_dir}")
    print(f"Training samples: {len(train_ids)} (vs {5471} current = +{(len(train_ids)/5471-1)*100:.0f}%)")
    print(f"Validation samples: {len(val_ids)} (vs {1328} current = +{(len(val_ids)/1328-1)*100:.0f}%)")
    print(f"Test samples: {len(test_ids)} (completely new holdout set)")
    print(f"Report saved: {report_path}")
    
    return report

def main():
    """Prepare full dataset for training"""
    
    print("PREPARING FULL DATASET FOR TRAINING")
    print("="*50)
    
    # Step 1: Analyze available data
    dataset_info = analyze_full_dataset()
    
    if dataset_info['total_matched'] < 1000:
        print("ERROR: Not enough matched samples found!")
        return
    
    # Step 2: Create segmentation masks
    created_masks = create_segmentation_masks()
    
    if created_masks < 1000:
        print("ERROR: Failed to create enough masks!")
        return
    
    # Step 3: Create train/val/test split
    train_ids, val_ids, test_ids = create_train_val_test_split(
        dataset_info['matched_ids']
    )
    
    # Step 4: Create MMSegmentation dataset
    base_dir = create_mmseg_dataset(train_ids, val_ids, test_ids)
    
    # Step 5: Generate report
    report = generate_dataset_report(base_dir, train_ids, val_ids, test_ids)
    
    print(f"\nNext steps:")
    print(f"1. Review dataset at: {base_dir}")
    print(f"2. Update training script to use new dataset path")
    print(f"3. Run training with {len(train_ids)} samples")
    print(f"4. Expect significant performance improvement!")
    
    return report

if __name__ == "__main__":
    report = main()