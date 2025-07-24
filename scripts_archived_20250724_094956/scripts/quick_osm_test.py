#!/usr/bin/env python3
"""
Quick test to download a small sample of OSM tiles.
"""

import os
import requests
import time
import math
import random
from pathlib import Path
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lat_lon_to_tile(lat, lon, zoom):
    """Convert latitude/longitude to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def download_tile(x, y, zoom, session, output_dir, tile_id):
    """Download a single tile."""
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        # Convert to RGB and resize
        img = Image.open(io.BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize((1280, 1280), Image.LANCZOS)
        
        # Save
        filename = f"osm_tile_{tile_id:06d}.jpg"
        filepath = output_dir / filename
        img_resized.save(filepath, "JPEG", quality=90)
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to download tile {x},{y}: {e}")
        return None

def main():
    output_dir = Path("data/unlabeled_aerial/osm_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'LaneSegNet Research Project (academic use)'
    })
    
    # London center
    lat, lon = 51.5074, -0.1278
    zoom = 17
    
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    
    # Download 20 tiles around London center
    successful = 0
    target = 20
    
    print(f"Downloading {target} test tiles from London...")
    
    for i in range(target):
        # Small offset around center
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        
        x = center_x + offset_x
        y = center_y + offset_y
        
        filepath = download_tile(x, y, zoom, session, output_dir, i)
        
        if filepath:
            successful += 1
            print(f"[{successful}/{target}] Downloaded: {Path(filepath).name}")
        
        # Rate limiting
        time.sleep(1)
    
    print(f"\n[SUCCESS] Downloaded {successful}/{target} test tiles")
    print(f"[INFO] Saved to: {output_dir}")
    
    if successful >= 10:
        print("[READY] OSM tile download is working - ready for full collection")
        return True
    else:
        print("[WARNING] OSM tile download had issues - check network connection")
        return False

if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test interrupted by user")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")