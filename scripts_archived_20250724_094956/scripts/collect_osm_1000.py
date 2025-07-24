#!/usr/bin/env python3
"""
Collect ~1000 OSM tiles for SSL pre-training demo.
Focused collection from multiple European cities.
"""

import os
import requests
import time
import math
import random
import json
from pathlib import Path
from PIL import Image
import io
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMCollector1000:
    def __init__(self, output_dir="data/unlabeled_aerial/osm_1000"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_count = 1000
        self.collected_count = 0
        
        # European cities with good road infrastructure
        self.cities = {
            'london': {'lat': 51.5074, 'lon': -0.1278, 'samples': 200},
            'paris': {'lat': 48.8566, 'lon': 2.3522, 'samples': 150},
            'berlin': {'lat': 52.5200, 'lon': 13.4050, 'samples': 150},
            'madrid': {'lat': 40.4168, 'lon': -3.7038, 'samples': 120},
            'rome': {'lat': 41.9028, 'lon': 12.4964, 'samples': 120},
            'amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'samples': 100},
            'barcelona': {'lat': 41.3851, 'lon': 2.1734, 'samples': 80},
            'vienna': {'lat': 48.2082, 'lon': 16.3738, 'samples': 80}
        }
        
        self.zoom = 17
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LaneSegNet Research Project (academic use)'
        })
        
    def lat_lon_to_tile(self, lat, lon, zoom):
        """Convert latitude/longitude to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    def download_tile(self, x, y, tile_id):
        """Download a single tile."""
        url = f"https://tile.openstreetmap.org/{self.zoom}/{x}/{y}.png"
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            # Convert to RGB and resize
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize((1280, 1280), Image.LANCZOS)
            
            # Save
            filename = f"osm_{tile_id:06d}.jpg"
            filepath = self.output_dir / filename
            img_resized.save(filepath, "JPEG", quality=90)
            
            return str(filepath)
            
        except Exception as e:
            logger.debug(f"Failed to download tile {x},{y}: {e}")
            return None
    
    def collect_city_tiles(self, city_name, city_info):
        """Collect tiles for a specific city."""
        center_x, center_y = self.lat_lon_to_tile(
            city_info['lat'], city_info['lon'], self.zoom
        )
        
        target_samples = city_info['samples']
        successful = 0
        attempts = 0
        max_attempts = target_samples * 3  # Allow for failures
        
        with tqdm(total=target_samples, desc=f"Collecting {city_name}") as pbar:
            while successful < target_samples and attempts < max_attempts:
                # Random offset within city bounds
                radius = 20  # Tile radius for city coverage
                offset_x = random.randint(-radius, radius)
                offset_y = random.randint(-radius, radius)
                
                x = center_x + offset_x
                y = center_y + offset_y
                
                # Check valid coordinates
                max_coord = 2 ** self.zoom
                if not (0 <= x < max_coord and 0 <= y < max_coord):
                    attempts += 1
                    continue
                
                filepath = self.download_tile(x, y, self.collected_count)
                attempts += 1
                
                if filepath:
                    successful += 1
                    self.collected_count += 1
                    pbar.update(1)
                
                # Rate limiting for OSM servers
                time.sleep(0.5)
                
                # Longer pause every 20 requests
                if attempts % 20 == 0:
                    time.sleep(2)
        
        logger.info(f"Collected {successful}/{target_samples} tiles from {city_name}")
        return successful
    
    def collect_all_cities(self):
        """Collect tiles from all cities."""
        logger.info(f"Starting collection of {self.target_count} OSM tiles...")
        
        total_collected = 0
        results = {}
        
        for city_name, city_info in self.cities.items():
            if self.collected_count >= self.target_count:
                break
                
            collected = self.collect_city_tiles(city_name, city_info)
            results[city_name] = {
                'collected': collected,
                'target': city_info['samples'],
                'rate': f"{collected/city_info['samples']*100:.1f}%"
            }
            total_collected += collected
            
            logger.info(f"Progress: {self.collected_count}/{self.target_count} total collected")
        
        return results, total_collected
    
    def create_manifest(self, results, total_collected):
        """Create manifest for collected tiles."""
        manifest = {
            'source': 'OpenStreetMap Tiles',
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_target': self.target_count,
            'total_collected': total_collected,
            'collection_rate': f"{total_collected/self.target_count*100:.1f}%",
            'zoom_level': self.zoom,
            'cities': results,
            'attribution': 'Map data (c) OpenStreetMap contributors',
            'usage': 'Academic research - self-supervised pre-training'
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest: {manifest_path}")
        return manifest

def main():
    collector = OSMCollector1000()
    
    print("OSM Tile Collection for SSL Pre-training")
    print("=" * 50)
    print(f"Target: {collector.target_count} tiles")
    print(f"Cities: {len(collector.cities)}")
    print(f"Output: {collector.output_dir}")
    print("=" * 50)
    
    try:
        results, total_collected = collector.collect_all_cities()
        manifest = collector.create_manifest(results, total_collected)
        
        print("\n" + "=" * 50)
        print("COLLECTION SUMMARY")
        print("=" * 50)
        
        for city, info in results.items():
            print(f"{city:<12}: {info['collected']:>3} / {info['target']:>3} ({info['rate']})")
        
        print("-" * 50)
        print(f"TOTAL: {total_collected:>3} / {collector.target_count:>3} ({manifest['collection_rate']})")
        print(f"FILES: {collector.output_dir}")
        
        if total_collected >= 500:
            print("\n[SUCCESS] Sufficient tiles collected for SSL demo!")
            print("[READY] Proceed with Self-Supervised pre-training")
        else:
            print(f"\n[PARTIAL] Only {total_collected} tiles collected")
            print("[INFO] Can still proceed with available data")
            
        return total_collected >= 500
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Collection stopped by user")
        print(f"[INFO] Collected {collector.collected_count} tiles before interruption")
    except Exception as e:
        print(f"\n[ERROR] Collection failed: {e}")
        logger.error(f"Collection failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    main()