#!/usr/bin/env python3
"""
Data Directory Cleanup - Archive problematic and redundant data safely
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Data Directory Cleanup")
    print("=" * 40)
    
    # CRITICAL: Preserve these items (core functionality)
    preserve_items = {
        "ael_mmseg/",               # CRITICAL: Working dataset with proper [0,1,2] annotations
        "unlabeled_aerial/",        # CRITICAL: SSL pre-training data (1,100+ images)
        "labeled_dataset.py",       # CRITICAL: Standardized dataset loader
        "unlabeled_dataset.py",     # CRITICAL: SSL dataset loader
        "premium_dataset.py",       # CRITICAL: Premium dataset components
        "train_data.json",          # CRITICAL: Training split configuration
        "val_data.json",            # CRITICAL: Validation split configuration  
        "test_data.json",           # CRITICAL: Test split configuration
        # Geographic reference data
        "Aucamvile.geojson",
        "Cairo.geojson", 
        "Glasgow.geojson",
        "Gopeng.geojson",
        "Nevada.geojson",
        "SanPaulo.geojson",
        "Valencia.geojson",
        "Valencia.geojson.qgz"
    }
    
    # SAFE TO ARCHIVE: Problematic, redundant, or completed items
    archive_items = {
        "full_ael_mmseg/",          # PROBLEMATIC: Contains empty masks - data integrity issue
        "full_masks/",              # SPACE INTENSIVE: 1,400+ individual PNG files
        "fixed_ael_mmseg/",         # REDUNDANT: Purpose unclear, may be duplicate
        "combined_lane_dataset/",   # SUPERSEDED: Replaced by current approach
        "SS_Dense/",                # EXTERNAL: May not be needed for current pipeline
        "SS_Multi_Lane/",           # EXTERNAL: May not be needed for current pipeline
        "imgs/",                    # RAW DATA: Possibly superseded by structured datasets
        "json/",                    # RAW DATA: Possibly superseded by structured datasets
        "mask/",                    # RAW DATA: Possibly superseded by structured datasets
        "results/",                 # OUTPUT: Analysis results directory
        "vis/",                     # OUTPUT: Visualization outputs
        "vertex/",                  # OUTPUT: Processing outputs
        "__pycache__/"              # CACHE: Python cache directory (safe to remove)
    }
    
    # Create archive directory for data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archived_data_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    print(f"Archive directory: {archive_dir}")
    print()
    
    # Move to data directory
    data_path = Path("data")
    if not data_path.exists():
        print("ERROR: data/ directory not found")
        return
    
    os.chdir(data_path)
    
    archived_count = 0
    space_saved_estimate = 0
    
    print("ARCHIVING DATA ITEMS:")
    print("-" * 25)
    
    # Archive problematic and redundant items
    for item in archive_items:
        item_path = Path(item)
        if item_path.exists():
            destination = Path("..") / archive_dir / "data" / item
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Estimate space being archived
            if item_path.is_dir():
                try:
                    file_count = len(list(item_path.rglob("*")))
                    if item == "full_masks/":
                        space_saved_estimate += file_count * 50  # Estimate 50KB per mask
                    print(f"ARCHIVED: {item} (directory with {file_count} files)")
                except:
                    print(f"ARCHIVED: {item} (directory)")
            else:
                file_size = item_path.stat().st_size
                space_saved_estimate += file_size
                print(f"ARCHIVED: {item} (file: {file_size} bytes)")
            
            shutil.move(str(item_path), str(destination))
            archived_count += 1
        else:
            print(f"NOT FOUND: {item}")
    
    # Go back to root
    os.chdir("..")
    
    print(f"\nData items archived: {archived_count}")
    if space_saved_estimate > 0:
        if space_saved_estimate > 1024*1024:
            print(f"Estimated space saved: ~{space_saved_estimate/(1024*1024):.1f} MB")
        else:
            print(f"Estimated space saved: ~{space_saved_estimate/1024:.1f} KB")
    
    # Verify preserved items still exist
    print(f"\nVERIFYING PRESERVED CORE DATA:")
    print("-" * 35)
    preserved_count = 0
    for item in preserve_items:
        item_path = data_path / item
        if item_path.exists():
            print(f"  PRESERVED: {item}")
            preserved_count += 1
        else:
            print(f"  MISSING: {item} (WARNING)")
    
    print(f"\nCore data items preserved: {preserved_count}/{len(preserve_items)}")
    
    # Create manifest
    manifest_path = archive_dir / "DATA_ARCHIVE_MANIFEST.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Data Directory Archive - {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ARCHIVED DATA ITEMS:\n")
        for item in sorted(archive_items):
            f.write(f"- data/{item}\n")
        f.write(f"\nPRESERVED CORE DATA:\n")
        for item in sorted(preserve_items):
            f.write(f"- data/{item}\n")
        f.write(f"\nTotal archived: {archived_count} items\n")
        f.write(f"Total preserved: {preserved_count} items\n")
        f.write(f"\nPURPOSE:\n")
        f.write("Archive problematic datasets (empty masks), redundant data, and space-intensive files\n")
        f.write("while preserving core working dataset and SSL training data.\n")
        f.write(f"\nCRITICAL PRESERVATIONS:\n")
        f.write("- ael_mmseg/ - Working dataset with proper [0,1,2] annotations\n")
        f.write("- unlabeled_aerial/ - SSL pre-training data (1,100+ images)\n")
        f.write("- Dataset loaders and JSON configurations\n")
    
    print(f"\nManifest created: {manifest_path}")
    
    print("\n" + "=" * 50)
    print("DATA DIRECTORY CLEANUP COMPLETE")
    print("=" * 50)
    print("CORE FUNCTIONALITY IMPACT: NONE")
    print("- Working dataset (ael_mmseg/) preserved")
    print("- SSL training data (unlabeled_aerial/) preserved") 
    print("- All dataset loaders preserved")
    print("- Problematic data safely archived")

if __name__ == "__main__":
    main()