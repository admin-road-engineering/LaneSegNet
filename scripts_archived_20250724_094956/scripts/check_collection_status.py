#!/usr/bin/env python3
"""
Check status of unlabeled data collection.
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_collection_status():
    """Check collection status across all sources."""
    base_dir = Path("data/unlabeled_aerial")
    
    if not base_dir.exists():
        print("[ERROR] Unlabeled data directory not found!")
        return
    
    # Sources to check
    sources = {
        'osm_1000': {'target': 1000, 'description': 'OpenStreetMap tiles'},
        'osm_test': {'target': 20, 'description': 'OpenStreetMap test samples'},
        'skyscapes/processed': {'target': 3000, 'description': 'SkyScapes aerial images'},
        'carla_synthetic/processed': {'target': 2000, 'description': 'CARLA synthetic scenes'},
        'cityscapes_aerial/processed': {'target': 1000, 'description': 'Cityscapes aerial transforms'},
        'consolidated': {'target': 7000, 'description': 'All consolidated images'}
    }
    
    total_found = 0
    total_target = 0
    
    print("\n" + "="*70)
    print("UNLABELED DATA COLLECTION STATUS")
    print("="*70)
    
    for source_path, info in sources.items():
        source_dir = base_dir / source_path
        
        if source_dir.exists():
            # Count images
            image_count = 0
            for ext in ['*.jpg', '*.png']:
                image_count += len(list(source_dir.glob(ext)))
            
            # Check for manifest
            manifest_files = list(source_dir.glob("*manifest.json"))
            has_manifest = "✓" if manifest_files else "✗"
            
            # Status
            if image_count >= info['target']:
                status = "[COMPLETE]"
            elif image_count > 0:
                status = "[PARTIAL] "
            else:
                status = "[EMPTY]   "
            
            progress = f"{image_count:>4} / {info['target']:<4}"
            percent = f"({image_count/info['target']*100:>5.1f}%)"
            
            print(f"{status} {source_path:<25} {progress} {percent} {has_manifest} {info['description']}")
            
            total_found += image_count
            total_target += info['target']
            
        else:
            print(f"[MISSING]  {source_path:<25} {'---':>4} / {info['target']:<4} (  0.0%) ✗ {info['description']}")
            total_target += info['target']
    
    print("-"*70)
    print(f"TOTALS: {total_found:>4} / {total_target:<4} ({total_found/total_target*100:>5.1f}%)")
    print("="*70)
    
    # SSL readiness assessment
    print("\nSELF-SUPERVISED LEARNING READINESS:")
    if total_found >= 5000:
        print("🟢 EXCELLENT (5k+) - Full SSL pre-training recommended")
    elif total_found >= 2000:
        print("🟡 GOOD (2k+) - SSL pre-training will be effective")
    elif total_found >= 1000:
        print("🟡 MODERATE (1k+) - SSL demo feasible")
    elif total_found >= 500:
        print("🟠 LIMITED (500+) - Basic SSL demo possible")
    else:
        print("🔴 INSUFFICIENT (<500) - More data collection needed")
    
    # Next steps
    print(f"\nRECOMMENDED NEXT STEPS:")
    if total_found < 1000:
        print("1. Complete OSM tile collection (run: python scripts/collect_osm_1000.py)")
        print("2. Download SkyScapes dataset (run: python scripts/download_skyscapes.py)")
    elif total_found < 3000:
        print("1. Download SkyScapes dataset for more diversity")
        print("2. Consider CARLA synthetic generation if available")
    else:
        print("1. Proceed with SSL pre-training implementation")
        print("2. Create Masked AutoEncoder for pre-training")
    
    print(f"\nCOLLECTION SCRIPTS:")
    print(f"- Full collection: run_data_collection.bat")
    print(f"- Individual sources: scripts/download_*.py, scripts/generate_*.py")
    print(f"- Consolidation: python scripts/consolidate_unlabeled_data.py")
    
    return total_found

def main():
    try:
        total = check_collection_status()
        return total >= 1000
    except Exception as e:
        print(f"\n[ERROR] Status check failed: {e}")
        logger.error(f"Status check failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    main()