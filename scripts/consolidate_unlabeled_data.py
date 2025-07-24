#!/usr/bin/env python3
"""
Consolidate all collected unlabeled data into unified structure.
"""

import os
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def consolidate_datasets():
    """Consolidate all unlabeled datasets."""
    base_dir = Path("data/unlabeled_aerial")
    consolidated_dir = base_dir / "consolidated"
    consolidated_dir.mkdir(exist_ok=True)
    
    # Source directories to consolidate
    sources = {
        'osm_1000': 'OpenStreetMap tiles',
        'osm_test': 'OpenStreetMap test samples',
        'skyscapes/processed': 'SkyScapes aerial images',
        'carla_synthetic/processed': 'CARLA synthetic scenes',
        'cityscapes_aerial/processed': 'Cityscapes aerial transforms'
    }
    
    consolidated_manifest = {
        'collection_date': str(datetime.now()),
        'sources': {},
        'total_images': 0,
        'consolidated_images': []
    }
    
    total_copied = 0
    
    for source_path, description in sources.items():
        source_dir = base_dir / source_path
        
        if source_dir.exists():
            # Find all images
            image_files = []
            for ext in ['*.jpg', '*.png']:
                image_files.extend(source_dir.glob(ext))
            
            logger.info(f"Found {len(image_files)} images in {source_path}")
            
            # Copy to consolidated directory
            copied_count = 0
            for img_file in image_files:
                try:
                    # Create unique filename
                    source_name = source_path.replace('/', '_').replace('\\', '_')
                    new_name = f"{source_name}_{img_file.name}"
                    dest_path = consolidated_dir / new_name
                    
                    if not dest_path.exists():
                        shutil.copy2(img_file, dest_path)
                        copied_count += 1
                        
                        consolidated_manifest['consolidated_images'].append({
                            'source': source_path,
                            'original_path': str(img_file),
                            'consolidated_path': str(dest_path),
                            'filename': new_name
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to copy {img_file}: {e}")
            
            consolidated_manifest['sources'][source_path] = {
                'description': description,
                'found': len(image_files),
                'copied': copied_count,
                'directory': str(source_dir)
            }
            
            total_copied += copied_count
            logger.info(f"Copied {copied_count} images from {source_path}")
        
        else:
            logger.info(f"Source directory not found: {source_dir}")
            consolidated_manifest['sources'][source_path] = {
                'description': description,
                'found': 0,
                'copied': 0,
                'directory': str(source_dir),
                'status': 'not_found'
            }
    
    consolidated_manifest['total_images'] = total_copied
    
    # Save manifest
    manifest_path = consolidated_dir / "consolidated_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(consolidated_manifest, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CONSOLIDATED UNLABELED DATA SUMMARY")
    print("="*60)
    
    for source, info in consolidated_manifest['sources'].items():
        status = "[FOUND]" if info['copied'] > 0 else "[MISSING]"
        print(f"{status} {source:<25}: {info['copied']:>4} images ({info['description']})")
    
    print("-"*60)
    print(f"TOTAL CONSOLIDATED: {total_copied:>4} images")
    print(f"LOCATION: {consolidated_dir}")
    print(f"MANIFEST: {manifest_path}")
    print("="*60)
    
    # SSL readiness check
    if total_copied >= 1000:
        print("\n[EXCELLENT] 1000+ images - Ready for full SSL pre-training!")
    elif total_copied >= 500:
        print("\n[GOOD] 500+ images - Ready for SSL demo!")
    else:
        print(f"\n[LIMITED] Only {total_copied} images - Consider collecting more")
    
    return total_copied

def main():
    try:
        import pandas as pd
    except ImportError:
        import datetime as dt
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return dt.datetime.now().isoformat()
    
    try:
        total = consolidate_datasets()
        if total > 0:
            print(f"\n[SUCCESS] Consolidated {total} unlabeled images")
        else:
            print("\n[WARNING] No images found to consolidate")
    except Exception as e:
        print(f"\n[ERROR] Consolidation failed: {e}")
        logger.error(f"Consolidation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()