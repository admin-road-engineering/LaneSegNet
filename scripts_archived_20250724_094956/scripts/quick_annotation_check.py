#!/usr/bin/env python3
"""
Quick Annotation Check
=====================

Quick check to see how many samples have lane annotations in the AEL dataset.
"""

import json
from pathlib import Path
import random

def quick_check():
    """Quick check of annotation status"""
    
    print("QUICK ANNOTATION CHECK")
    print("=" * 25)
    
    json_dir = Path('data/json')
    json_files = list(json_dir.glob('*.json'))
    
    # Sample 100 random files for quick assessment
    sample_size = min(100, len(json_files))
    sample_files = random.sample(json_files, sample_size)
    
    print(f"Checking {sample_size} random files from {len(json_files)} total...")
    
    annotated = 0
    empty = 0
    total_lanes = 0
    
    for json_file in sample_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            lanes = data.get('lanes', [])
            if lanes:
                annotated += 1
                total_lanes += len(lanes)
            else:
                empty += 1
                
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            empty += 1
    
    print(f"\nQUICK RESULTS:")
    print(f"  Files with lanes: {annotated}/{sample_size} ({annotated/sample_size*100:.1f}%)")
    print(f"  Files without lanes: {empty}/{sample_size} ({empty/sample_size*100:.1f}%)")
    print(f"  Average lanes per annotated file: {total_lanes/max(annotated,1):.1f}")
    
    # Extrapolate to full dataset
    if annotated > 0:
        estimated_annotated = int(len(json_files) * annotated / sample_size)
        print(f"\nESTIMATED FULL DATASET:")
        print(f"  Estimated annotated samples: {estimated_annotated}")
        print(f"  Estimated empty samples: {len(json_files) - estimated_annotated}")
    
    return annotated, empty

if __name__ == "__main__":
    quick_check()