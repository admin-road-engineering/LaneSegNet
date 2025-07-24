#!/usr/bin/env python3
"""
Download OpenStreetMap aerial tiles for unlabeled imagery.
Target: ~5k aerial road tiles for self-supervised pre-training.
"""

import os
import requests
import time
import json
from pathlib import Path
from PIL import Image
import io
import logging
from tqdm import tqdm
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMTileDownloader:
    def __init__(self, download_dir="data/unlabeled_aerial/osm_tiles"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenStreetMap tile servers (free, no API key required)
        self.tile_servers = [
            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", 
            "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"
        ]
        
        # Major cities with good road infrastructure
        self.target_cities = {
            'london': {'lat': 51.5074, 'lon': -0.1278, 'radius': 0.2},
            'paris': {'lat': 48.8566, 'lon': 2.3522, 'radius': 0.15},
            'berlin': {'lat': 52.5200, 'lon': 13.4050, 'radius': 0.15},
            'madrid': {'lat': 40.4168, 'lon': -3.7038, 'radius': 0.15},
            'rome': {'lat': 41.9028, 'lon': 12.4964, 'radius': 0.15},
            'amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'radius': 0.1},
            'barcelona': {'lat': 41.3851, 'lon': 2.1734, 'radius': 0.1},
            'munich': {'lat': 48.1351, 'lon': 11.5820, 'radius': 0.1},
            'vienna': {'lat': 48.2082, 'lon': 16.3738, 'radius': 0.1},
            'zurich': {'lat': 47.3769, 'lon': 8.5417, 'radius': 0.08}
        }
        
        self.zoom_level = 17  # Good balance: detail vs coverage
        self.target_count = 5000
        self.collected_count = 0
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
    
    def generate_tile_coords(self, city_info, num_tiles=500):
        """Generate tile coordinates around a city center."""
        center_lat = city_info['lat']
        center_lon = city_info['lon'] 
        radius = city_info['radius']
        
        center_x, center_y = self.lat_lon_to_tile(center_lat, center_lon, self.zoom_level)
        
        # Calculate tile radius (approximate)
        tile_radius = int(radius * 2 ** self.zoom_level / 360 * 256)
        
        tiles = []
        for _ in range(num_tiles):
            # Random offset within radius
            offset_x = random.randint(-tile_radius, tile_radius)
            offset_y = random.randint(-tile_radius, tile_radius)
            
            x = center_x + offset_x
            y = center_y + offset_y
            
            # Ensure coordinates are valid
            max_coord = 2 ** self.zoom_level
            if 0 <= x < max_coord and 0 <= y < max_coord:
                tiles.append((x, y))
        
        return tiles
    
    def download_tile(self, x, y, tile_id):
        """Download a single tile."""
        server_url = random.choice(self.tile_servers)
        url = server_url.format(z=self.zoom_level, x=x, y=y)
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to RGB and resize to match our training data
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 1280x1280 to match training data
            img_resized = img.resize((1280, 1280), Image.LANCZOS)
            
            # Save image
            filename = f"osm_tile_{tile_id:06d}_{x}_{y}.jpg"
            filepath = self.download_dir / "processed" / filename
            filepath.parent.mkdir(exist_ok=True)
            
            img_resized.save(filepath, "JPEG", quality=90)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download tile {x},{y}: {e}")
            return None
    
    def is_road_rich_tile(self, image_path):
        """
        Heuristic to determine if tile contains significant road infrastructure.
        This is a simplified check - in practice you'd want more sophisticated analysis.
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Roads typically appear as:
            # 1. Linear structures (high edge density)
            # 2. Gray/dark colors
            # 3. Connected networks
            
            # Simple edge detection
            from scipy import ndimage
            edges = ndimage.sobel(gray)
            edge_density = np.mean(edges > 20)
            
            # Check for road-like colors (gray areas)
            gray_mask = (img_array[:,:,0] > 80) & (img_array[:,:,0] < 150) & \
                       (abs(img_array[:,:,0] - img_array[:,:,1]) < 20) & \
                       (abs(img_array[:,:,1] - img_array[:,:,2]) < 20)
            gray_ratio = np.mean(gray_mask)
            
            # Tile is "road-rich" if it has both edges and gray areas
            return edge_density > 0.1 and gray_ratio > 0.05
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            return False
    
    def download_city_tiles(self, city_name, city_info, tiles_per_city=500):
        """Download tiles for a specific city."""
        logger.info(f"Downloading tiles for {city_name}...")
        
        tile_coords = self.generate_tile_coords(city_info, tiles_per_city)
        successful_downloads = []
        
        for i, (x, y) in enumerate(tqdm(tile_coords, desc=f"Downloading {city_name} tiles")):
            if self.collected_count >= self.target_count:
                break
                
            tile_id = self.collected_count
            filepath = self.download_tile(x, y, tile_id)
            
            if filepath:
                # Check if tile contains roads
                if self.is_road_rich_tile(filepath):
                    successful_downloads.append(filepath)
                    self.collected_count += 1
                else:
                    # Remove non-road tiles
                    os.remove(filepath)
            
            # Rate limiting - OSM tile servers have usage policies
            time.sleep(0.1)  # 100ms delay between requests
            
            # Longer delay every 50 requests
            if i % 50 == 0 and i > 0:
                time.sleep(2)
        
        logger.info(f"Collected {len(successful_downloads)} road-rich tiles from {city_name}")
        return successful_downloads
    
    def create_manifest(self, all_images):
        """Create manifest file for downloaded tiles."""
        manifest = {
            'source': 'OpenStreetMap Tiles',
            'count': len(all_images),
            'target_count': self.target_count,
            'zoom_level': self.zoom_level,
            'cities': list(self.target_cities.keys()),
            'images': all_images,
            'processing_date': str(pd.Timestamp.now()),
            'description': 'Road-rich aerial tiles from OpenStreetMap for SSL pre-training',
            'attribution': 'Map data © OpenStreetMap contributors'
        }
        
        manifest_path = self.download_dir / "osm_tiles_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest: {manifest_path}")
    
    def download_all_cities(self):
        """Download tiles from all target cities."""
        logger.info("Starting OpenStreetMap tile download...")
        
        all_images = []
        tiles_per_city = max(50, self.target_count // len(self.target_cities))
        
        for city_name, city_info in self.target_cities.items():
            if self.collected_count >= self.target_count:
                break
                
            city_images = self.download_city_tiles(city_name, city_info, tiles_per_city)
            all_images.extend(city_images)
            
            logger.info(f"Progress: {self.collected_count}/{self.target_count} tiles collected")
        
        if all_images:
            self.create_manifest(all_images)
            logger.info(f"OSM download complete: {len(all_images)}/{self.target_count} tiles")
        else:
            logger.warning("No road-rich tiles collected from OSM")
        
        return all_images

def main():
    """Main execution function."""
    # Import numpy and scipy if available
    try:
        import numpy as np
        import scipy
        globals()['np'] = np
    except ImportError:
        print("[ERROR] NumPy and SciPy required for road detection. Please install: pip install numpy scipy")
        return
    
    downloader = OSMTileDownloader()
    
    try:
        collected_images = downloader.download_all_cities()
        
        if collected_images:
            print(f"[SUCCESS] Successfully collected {len(collected_images)} road-rich tiles from OSM")
            print(f"[INFO] Images saved to: {downloader.download_dir}/processed/")
            print(f"[INFO] Manifest: {downloader.download_dir}/osm_tiles_manifest.json")
            print(f"[INFO] Attribution: Map data (c) OpenStreetMap contributors")
        else:
            print("[ERROR] Failed to collect tiles from OpenStreetMap")
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Download interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

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