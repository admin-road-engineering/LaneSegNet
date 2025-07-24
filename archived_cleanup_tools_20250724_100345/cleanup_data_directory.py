#!/usr/bin/env python3
"""
Data Directory Analysis - Identify obsolete/problematic data that can be archived
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Data Directory Analysis")
    print("=" * 45)
    
    # KEEP these data components (critical for production)
    keep_items = {
        # Current working dataset with proper annotations
        "ael_mmseg/",                # CURRENT: Working dataset with proper [0,1,2] annotations
        
        # SSL pre-training data
        "unlabeled_aerial/",         # CURRENT: 1,100+ images for SSL pre-training (if exists)
        
        # Dataset loaders and configurations
        "labeled_dataset.py",        # CURRENT: Standardized dataset loader
        "unlabeled_dataset.py",      # CURRENT: SSL dataset loader
        "train_data.json",           # CURRENT: Training split configuration
        "val_data.json",             # CURRENT: Validation split configuration
        "test_data.json",            # CURRENT: Test split configuration
        
        # Premium dataset components (if needed)
        "premium_dataset.py",        # CURRENT: Premium dataset loader (may be needed)
        
        # Geographic data files
        "Aucamvile.geojson",        # REFERENCE: Geographic boundaries
        "Cairo.geojson",            # REFERENCE: Geographic boundaries
        "Glasgow.geojson",          # REFERENCE: Geographic boundaries
        "Gopeng.geojson",           # REFERENCE: Geographic boundaries
        "Nevada.geojson",           # REFERENCE: Geographic boundaries
        "SanPaulo.geojson",         # REFERENCE: Geographic boundaries
        "Valencia.geojson",         # REFERENCE: Geographic boundaries
        "Valencia.geojson.qgz"      # REFERENCE: Geographic data
    }
    
    # PROBLEMATIC items that should be investigated/archived
    problematic_items = {
        "full_ael_mmseg/",          # PROBLEMATIC: Contains empty masks - data integrity issue
        "fixed_ael_mmseg/",         # UNKNOWN: Status unknown - may be redundant
        "combined_lane_dataset/",   # HISTORICAL: May be superseded by current datasets
        "full_masks/",              # MASSIVE: 1,400+ individual PNG files taking up space
        "SS_Dense/",                # EXTERNAL: May be external dataset (if exists)
        "SS_Multi_Lane/",           # EXTERNAL: May be external dataset (if exists)
    }
    
    # Check data directory
    data_path = Path("data")
    if not data_path.exists():
        print("ERROR: data/ directory not found")
        return
    
    print(f"Analyzing data directory contents...")
    print()
    
    # Check what actually exists
    existing_items = []
    for item in data_path.iterdir():
        existing_items.append(item.name)
    
    print("CURRENT DATA DIRECTORY CONTENTS:")
    print("-" * 40)
    
    keep_found = []
    problematic_found = []
    unknown_found = []
    
    for item in sorted(existing_items):
        if item in keep_items:
            print(f"  KEEP: {item}")
            keep_found.append(item)
        elif item in problematic_items:
            print(f"  PROBLEMATIC: {item}")
            problematic_found.append(item)
        else:
            print(f"  UNKNOWN: {item}")
            unknown_found.append(item)
    
    print()
    print("ANALYSIS SUMMARY:")
    print("-" * 20)
    print(f"Items to KEEP: {len(keep_found)}")
    print(f"PROBLEMATIC items: {len(problematic_found)}")
    print(f"UNKNOWN items: {len(unknown_found)}")
    
    if problematic_found:
        print()
        print("PROBLEMATIC ITEMS DETAILS:")
        print("-" * 30)
        
        for item in problematic_found:
            item_path = data_path / item
            if item_path.exists():
                if item_path.is_dir():
                    try:
                        file_count = len(list(item_path.iterdir()))
                        print(f"  {item}: Directory with {file_count} files")
                        
                        # Special analysis for full_masks
                        if item == "full_masks/":
                            print(f"    ^ This directory contains {file_count} individual PNG mask files")
                            print(f"    ^ May be taking significant disk space")
                            
                        # Special analysis for full_ael_mmseg
                        if item == "full_ael_mmseg/":
                            print(f"    ^ This dataset was found to have empty masks (data integrity issue)")
                            print(f"    ^ Should be investigated or archived")
                            
                    except Exception as e:
                        print(f"  {item}: Directory (unable to count files: {e})")
                else:
                    file_size = item_path.stat().st_size
                    print(f"  {item}: File ({file_size} bytes)")
    
    if unknown_found:
        print()
        print("UNKNOWN ITEMS (need investigation):")
        print("-" * 35)
        for item in unknown_found:
            print(f"  {item}")
    
    print()
    print("RECOMMENDATIONS:")
    print("-" * 20)
    print("1. ARCHIVE full_ael_mmseg/ - Contains empty masks, data integrity compromised")
    print("2. INVESTIGATE fixed_ael_mmseg/ - Purpose unclear, may be redundant")
    print("3. CONSIDER archiving full_masks/ - 1,400+ individual files, space intensive")
    print("4. REVIEW combined_lane_dataset/ - May be superseded by current approach")
    print("5. PRESERVE ael_mmseg/ - This is the working dataset with proper annotations")
    print("6. PRESERVE unlabeled_aerial/ - Critical for SSL pre-training")
    
    # Analysis complete
    
    print()
    print("=" * 50)
    print("DATA DIRECTORY ANALYSIS COMPLETE")
    print("=" * 50)
    print("No changes made - analysis only.")
    print("Review recommendations before proceeding with cleanup.")

if __name__ == "__main__":
    main()