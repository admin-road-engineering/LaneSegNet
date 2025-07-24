#!/usr/bin/env python3
"""
Investigate Test Annotations
===========================

Check what's actually in the holdout test annotations to understand
why they all show 0 lane pixels.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json

def investigate_test_annotations():
    """Investigate the holdout test annotations in detail"""
    
    print("INVESTIGATING HOLDOUT TEST ANNOTATIONS")
    print("=" * 50)
    
    test_img_dir = Path('data/full_ael_mmseg/img_dir/test')
    test_ann_dir = Path('data/full_ael_mmseg/ann_dir/test')
    
    # Check if directories exist
    if not test_img_dir.exists():
        print(f"ERROR: {test_img_dir} does not exist")
        return
    if not test_ann_dir.exists():
        print(f"ERROR: {test_ann_dir} does not exist")
        return
    
    # Get all files
    img_files = list(test_img_dir.glob('*.jpg'))
    ann_files = list(test_ann_dir.glob('*.png'))
    
    print(f"Test images: {len(img_files)}")
    print(f"Test annotations: {len(ann_files)}")
    
    # Sample random files for investigation
    sample_count = min(20, len(ann_files))
    sample_files = random.sample(ann_files, sample_count)
    
    print(f"\nInvestigating {sample_count} random annotation files...")
    
    annotation_stats = {
        'all_zero': 0,
        'has_lanes': 0,
        'class_distributions': [],
        'file_sizes': [],
        'unique_values': set()
    }
    
    for i, ann_file in enumerate(sample_files):
        try:
            # Load annotation
            mask = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"  {ann_file.name}: Failed to load")
                continue
            
            # Analyze mask content
            unique_values = np.unique(mask)
            class_counts = np.bincount(mask.flatten())
            
            # Store statistics
            annotation_stats['unique_values'].update(unique_values.tolist())
            annotation_stats['file_sizes'].append(mask.size)
            annotation_stats['class_distributions'].append(class_counts)
            
            # Check if all zeros
            if np.all(mask == 0):
                annotation_stats['all_zero'] += 1
                print(f"  {ann_file.name}: ALL ZEROS (size: {mask.shape})")
            else:
                annotation_stats['has_lanes'] += 1
                lane_pixels = np.sum(mask > 0)
                print(f"  {ann_file.name}: {lane_pixels} lane pixels, classes: {unique_values}")
                
                # Show detailed breakdown for first few with lanes
                if annotation_stats['has_lanes'] <= 3:
                    for class_id in unique_values:
                        count = (mask == class_id).sum()
                        print(f"    Class {class_id}: {count} pixels ({count/mask.size*100:.1f}%)")
        
        except Exception as e:
            print(f"  {ann_file.name}: ERROR - {e}")
    
    print(f"\nANNOTATION ANALYSIS SUMMARY:")
    print(f"  Files analyzed: {sample_count}")
    print(f"  All-zero annotations: {annotation_stats['all_zero']}")
    print(f"  Annotations with lanes: {annotation_stats['has_lanes']}")
    print(f"  Unique pixel values found: {sorted(annotation_stats['unique_values'])}")
    
    # Check source JSON files to see if they contain lane data
    print(f"\nCHECKING SOURCE JSON ANNOTATIONS...")
    
    # Get corresponding JSON files for same samples
    json_dir = Path('data/json')
    json_stats = {
        'found': 0,
        'has_shapes': 0,
        'total_shapes': 0
    }
    
    for ann_file in sample_files[:10]:  # Check first 10
        sample_id = ann_file.stem
        json_file = json_dir / f"{sample_id}.json"
        
        if json_file.exists():
            json_stats['found'] += 1
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                shapes = data.get('shapes', [])
                if shapes:
                    json_stats['has_shapes'] += 1
                    json_stats['total_shapes'] += len(shapes)
                    print(f"  {sample_id}.json: {len(shapes)} shapes")
                    
                    # Show first shape as example
                    if len(shapes) > 0:
                        shape = shapes[0]
                        label = shape.get('label', 'unknown')
                        points = len(shape.get('points', []))
                        print(f"    Example: {label} with {points} points")
                else:
                    print(f"  {sample_id}.json: NO SHAPES")
                    
            except Exception as e:
                print(f"  {sample_id}.json: ERROR reading - {e}")
        else:
            print(f"  {sample_id}.json: NOT FOUND")
    
    print(f"\nJSON ANALYSIS:")
    print(f"  JSON files found: {json_stats['found']}/10")
    print(f"  JSON files with shapes: {json_stats['has_shapes']}")
    print(f"  Total shapes found: {json_stats['total_shapes']}")
    
    # Final diagnosis
    print(f"\nDIAGNOSIS:")
    
    if annotation_stats['all_zero'] == sample_count:
        print("  PROBLEM: ALL test annotations are empty!")
        print("  -> Issue with mask generation process")
        
        if json_stats['has_shapes'] > 0:
            print("  -> Source JSON files contain lane data")
            print("  -> Mask conversion process failed")
        else:
            print("  -> Source JSON files also empty")
            print("  -> Dataset split may have selected images without annotations")
    
    elif annotation_stats['has_lanes'] > 0:
        print("  MIXED: Some test annotations have lane data")
        print(f"  -> {annotation_stats['has_lanes']}/{sample_count} files have lanes")
        print("  -> May need larger sample to verify")
    
    else:
        print("  UNCLEAR: Need more investigation")
    
    return annotation_stats

def compare_with_training_annotations():
    """Compare test annotations with training annotations for reference"""
    
    print(f"\n" + "="*50)
    print("COMPARING WITH TRAINING ANNOTATIONS")
    print("="*50)
    
    train_ann_dir = Path('data/ael_mmseg/ann_dir/train')
    
    if not train_ann_dir.exists():
        print("Training annotations not found")
        return
    
    # Sample training annotations
    train_files = list(train_ann_dir.glob('*.png'))
    sample_train = random.sample(train_files, min(10, len(train_files)))
    
    train_with_lanes = 0
    
    for ann_file in sample_train:
        try:
            mask = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask > 0):
                train_with_lanes += 1
                lane_pixels = np.sum(mask > 0)
                print(f"  {ann_file.name}: {lane_pixels} lane pixels")
            else:
                print(f"  {ann_file.name}: ALL ZEROS")
        except Exception as e:
            print(f"  {ann_file.name}: ERROR - {e}")
    
    print(f"\nTraining annotations with lanes: {train_with_lanes}/{len(sample_train)}")
    
    return train_with_lanes

def main():
    """Run full investigation"""
    
    test_stats = investigate_test_annotations()
    train_lanes = compare_with_training_annotations()
    
    print(f"\n" + "="*50)
    print("INVESTIGATION CONCLUSION")
    print("="*50)
    
    if test_stats and test_stats['all_zero'] > test_stats['has_lanes']:
        if train_lanes > 0:
            print("CONFIRMED: Test annotation generation failed")
            print("-> Training annotations have lanes, test annotations are empty")
            print("-> Need to regenerate test masks from JSON files")
        else:
            print("DATASET ISSUE: Both test and training annotations problematic")
    else:
        print("Need more investigation - results unclear")

if __name__ == "__main__":
    main()