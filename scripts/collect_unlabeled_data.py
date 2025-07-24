#!/usr/bin/env python3
"""
Master script to collect unlabeled aerial imagery from all sources.
Coordinates collection from SkyScapes, OSM, CARLA, and Cityscapes.
Target: 15-20k unlabeled aerial images for self-supervised pre-training.
"""

import os
import json
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnlabeledDataCollector:
    def __init__(self, base_dir="data/unlabeled_aerial"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection targets per source
        self.collection_targets = {
            'skyscapes': {'target': 3000, 'priority': 1, 'script': 'download_skyscapes.py'},
            'osm_tiles': {'target': 5000, 'priority': 1, 'script': 'download_osm_tiles.py'},
            'carla_synthetic': {'target': 2000, 'priority': 2, 'script': 'generate_carla_aerial.py'},
            'cityscapes_aerial': {'target': 1000, 'priority': 2, 'script': 'transform_cityscapes_aerial.py'},
        }
        
        self.total_target = sum(source['target'] for source in self.collection_targets.values())
        self.collected_summary = {}
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        required_packages = {
            'requests': 'HTTP requests for downloading',
            'PIL': 'Image processing',
            'numpy': 'Numerical operations',
            'tqdm': 'Progress bars',
            'opencv-python': 'Computer vision operations'
        }
        
        missing_packages = []
        
        for package, description in required_packages.items():
            try:
                if package == 'PIL':
                    import PIL
                elif package == 'opencv-python':
                    import cv2
                else:
                    __import__(package)
                logger.debug(f"✅ {package}: Available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package}: Missing - {description}")
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def run_collection_script(self, source_name, source_info):
        """Run individual collection script."""
        script_path = Path("scripts") / source_info['script']
        
        if not script_path.exists():
            logger.error(f"Collection script not found: {script_path}")
            return {'source': source_name, 'success': False, 'count': 0, 'error': 'Script not found'}
        
        try:
            logger.info(f"Starting collection from {source_name}...")
            
            # Run collection script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Parse output for collection count
                output_lines = result.stdout.split('\n')
                count = 0
                
                for line in output_lines:
                    if 'Successfully collected' in line or 'Successfully generated' in line or 'Successfully transformed' in line:
                        # Extract number from success message
                        import re
                        numbers = re.findall(r'(\d+)', line)
                        if numbers:
                            count = int(numbers[0])
                            break
                
                logger.info(f"✅ {source_name}: Collected {count} images")
                return {
                    'source': source_name,
                    'success': True,
                    'count': count,
                    'target': source_info['target'],
                    'output': result.stdout
                }
            else:
                logger.error(f"❌ {source_name}: Failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return {
                    'source': source_name,
                    'success': False,
                    'count': 0,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {source_name}: Timeout after 1 hour")
            return {'source': source_name, 'success': False, 'count': 0, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"❌ {source_name}: Unexpected error: {e}")
            return {'source': source_name, 'success': False, 'count': 0, 'error': str(e)}
    
    def collect_parallel(self, max_workers=2):
        """Collect data from multiple sources in parallel."""
        logger.info(f"Starting parallel collection from {len(self.collection_targets)} sources...")
        
        # Sort sources by priority
        sorted_sources = sorted(
            self.collection_targets.items(),
            key=lambda x: x[1]['priority']
        )
        
        results = []
        
        # Run high priority sources first (parallel)
        high_priority = [(name, info) for name, info in sorted_sources if info['priority'] == 1]
        low_priority = [(name, info) for name, info in sorted_sources if info['priority'] == 2]
        
        # Collect high priority sources in parallel
        if high_priority:
            logger.info("Collecting high priority sources (SkyScapes, OSM)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_source = {
                    executor.submit(self.run_collection_script, name, info): name
                    for name, info in high_priority
                }
                
                for future in as_completed(future_to_source):
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result['source']}")
        
        # Collect low priority sources sequentially (resource intensive)
        if low_priority:
            logger.info("Collecting low priority sources (CARLA, Cityscapes)...")
            for name, info in low_priority:
                result = self.run_collection_script(name, info)
                results.append(result)
                logger.info(f"Completed: {result['source']}")
        
        return results
    
    def collect_sequential(self):
        """Collect data from sources sequentially."""
        logger.info("Starting sequential collection...")
        
        results = []
        
        # Sort by priority
        sorted_sources = sorted(
            self.collection_targets.items(),
            key=lambda x: x[1]['priority']
        )
        
        for source_name, source_info in sorted_sources:
            result = self.run_collection_script(source_name, source_info)
            results.append(result)
            
            # Short break between collections
            time.sleep(2)
        
        return results
    
    def consolidate_datasets(self, results):
        """Consolidate all collected datasets into unified structure."""
        consolidated_dir = self.base_dir / "consolidated"
        consolidated_dir.mkdir(exist_ok=True)
        
        all_images = []
        total_collected = 0
        
        for result in results:
            if result['success']:
                source_name = result['source']
                source_dir = self.base_dir / source_name / "processed"
                
                if source_dir.exists():
                    # Find all images in source directory
                    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
                    
                    for img_file in image_files:
                        # Create symbolic link in consolidated directory
                        link_name = f"{source_name}_{img_file.name}"
                        link_path = consolidated_dir / link_name
                        
                        try:
                            if not link_path.exists():
                                # Copy file instead of symlink for Windows compatibility
                                import shutil
                                shutil.copy2(img_file, link_path)
                            
                            all_images.append({
                                'source': source_name,
                                'original_path': str(img_file),
                                'consolidated_path': str(link_path)
                            })
                            
                        except Exception as e:
                            logger.warning(f"Failed to consolidate {img_file}: {e}")
                
                total_collected += result.get('count', 0)
        
        logger.info(f"Consolidated {len(all_images)} images from {total_collected} collected")
        return all_images, total_collected
    
    def create_master_manifest(self, results, all_images, total_collected):
        """Create master manifest for all collected data."""
        manifest = {
            'collection_date': str(pd.Timestamp.now()),
            'total_target': self.total_target,
            'total_collected': total_collected,
            'collection_rate': f"{total_collected/self.total_target*100:.1f}%",
            'sources': {},
            'consolidated_images': all_images,
            'summary': {
                'successful_sources': sum(1 for r in results if r['success']),
                'failed_sources': sum(1 for r in results if not r['success']),
                'ready_for_ssl': total_collected >= 10000  # Minimum for effective SSL
            }
        }
        
        # Add per-source details
        for result in results:
            manifest['sources'][result['source']] = {
                'success': result['success'],
                'collected': result.get('count', 0),
                'target': self.collection_targets[result['source']]['target'],
                'rate': f"{result.get('count', 0)/self.collection_targets[result['source']]['target']*100:.1f}%",
                'error': result.get('error', None) if not result['success'] else None
            }
        
        # Save manifest
        manifest_path = self.base_dir / "master_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created master manifest: {manifest_path}")
        return manifest
    
    def collect_all_sources(self, parallel=True):
        """Main method to collect from all sources."""
        if not self.check_dependencies():
            return False
        
        logger.info(f"🚀 Starting unlabeled data collection (Target: {self.total_target:,} images)")
        
        # Collect data
        if parallel:
            results = self.collect_parallel()
        else:
            results = self.collect_sequential()
        
        # Consolidate datasets
        all_images, total_collected = self.consolidate_datasets(results)
        
        # Create master manifest
        manifest = self.create_master_manifest(results, all_images, total_collected)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 UNLABELED DATA COLLECTION SUMMARY")
        print("="*60)
        
        for source, info in manifest['sources'].items():
            status = "✅" if info['success'] else "❌"
            print(f"{status} {source:<20}: {info['collected']:>5,} / {info['target']:>5,} ({info['rate']})")
        
        print("-"*60)
        print(f"📈 TOTAL COLLECTED: {total_collected:,} / {self.total_target:,} ({manifest['collection_rate']})")
        print(f"🎯 SSL READY: {'✅ YES' if manifest['summary']['ready_for_ssl'] else '❌ NO (need 10k+ images)'}")
        print(f"📁 CONSOLIDATED: {self.base_dir}/consolidated/")
        print(f"📄 MANIFEST: {self.base_dir}/master_manifest.json")
        print("="*60)
        
        return total_collected >= 10000

def main():
    """Main execution function."""
    collector = UnlabeledDataCollector()
    
    try:
        # Run collection
        success = collector.collect_all_sources(parallel=True)
        
        if success:
            print("\n🎉 Unlabeled data collection SUCCESSFUL!")
            print("✅ Ready to proceed with Self-Supervised pre-training")
        else:
            print("\n⚠️ Unlabeled data collection PARTIAL")
            print("💡 You can still proceed with available data")
            
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logger.error(f"Collection failed: {e}", exc_info=True)

if __name__ == "__main__":
    # Add pandas import for timestamp
    try:
        import pandas as pd
    except ImportError:
        import datetime as dt
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return dt.datetime.now().isoformat()
    
    main()