#!/usr/bin/env python3
"""
Check SSL Pre-training Readiness
Quick verification of unlabeled data availability without PyTorch dependencies.
"""

import os
import json
from pathlib import Path

def check_ssl_readiness():
    """Check if we're ready for SSL pre-training."""
    print("=" * 60)
    print("SSL PRE-TRAINING READINESS CHECK")
    print("=" * 60)
    
    base_dir = Path("data/unlabeled_aerial")
    
    # Check consolidated data
    consolidated_dir = base_dir / "consolidated"
    if consolidated_dir.exists():
        consolidated_images = len([f for f in consolidated_dir.glob("*.jpg")] + 
                                [f for f in consolidated_dir.glob("*.png")])
        print(f"[OK] Consolidated dataset: {consolidated_images} images")
        
        # Load manifest if available
        manifest_path = consolidated_dir / "consolidated_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"[INFO] Collection date: {manifest.get('collection_date', 'Unknown')}")
            print(f"[INFO] Sources available:")
            for source, info in manifest.get('sources', {}).items():
                count = info.get('images_copied', 0)
                if count > 0:
                    print(f"   - {source}: {count} images ({info.get('description', 'No description')})")
    else:
        print("[ERROR] Consolidated dataset not found")
        consolidated_images = 0
    
    # Check individual sources
    print(f"\n[INFO] INDIVIDUAL SOURCES:")
    sources = {
        'osm_1000': 'OpenStreetMap tiles (1000 target)',
        'osm_test': 'OpenStreetMap test samples', 
        'cityscapes_aerial/processed': 'Cityscapes aerial transforms',
        'skyscapes/processed': 'SkyScapes aerial images',
        'carla_synthetic/processed': 'CARLA synthetic scenes'
    }
    
    total_individual = 0
    for source_path, description in sources.items():
        source_dir = base_dir / source_path
        if source_dir.exists():
            count = len(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")))
            status = "[OK]" if count > 0 else "[WARN]"
            print(f"{status} {source_path}: {count} images")
            total_individual += count
        else:
            print(f"[MISSING] {source_path}: Not found")
    
    print(f"\n[SUMMARY]:")
    print(f"   Consolidated: {consolidated_images} images")
    print(f"   Individual sources: {total_individual} images")
    
    # SSL readiness assessment
    max_available = max(consolidated_images, total_individual)
    
    print(f"\n[SSL PRE-TRAINING ASSESSMENT]:")
    if max_available >= 1000:
        print(f"[EXCELLENT] {max_available} images - Full SSL pre-training recommended")
        recommendation = "excellent"
    elif max_available >= 500:
        print(f"[GOOD] {max_available} images - SSL pre-training will be effective")
        recommendation = "good"
    elif max_available >= 100:
        print(f"[LIMITED] {max_available} images - Basic SSL demo possible")
        recommendation = "limited"
    else:
        print(f"[INSUFFICIENT] {max_available} images - More data needed")
        recommendation = "insufficient"
    
    print(f"\n[NEXT STEPS]:")
    if recommendation in ['excellent', 'good']:
        print("1. Activate virtual environment: .venv\\Scripts\\activate.bat")
        print("2. Run SSL pre-training: python scripts/run_ssl_pretraining.py --epochs 50 --batch-size 16")
        print("3. Expected training time: 2-4 hours for 50 epochs")
        print("4. Expected improvement: +5-15% mIoU on downstream task")
    elif recommendation == 'limited':
        print("1. Consider collecting more data for better results")
        print("2. Or proceed with current data for demonstration")
        print("3. Reduce epochs to 25-30 for smaller dataset")
    else:
        print("1. Run data collection again: run_data_collection.bat")
        print("2. Ensure internet connectivity for OSM tile downloads")
    
    print("=" * 60)
    
    return max_available, recommendation

if __name__ == "__main__":
    check_ssl_readiness()