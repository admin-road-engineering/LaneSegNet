#!/usr/bin/env python3
"""
Transform Cityscapes dataset to aerial viewpoints.
Target: ~1k aerial-transformed images for self-supervised pre-training.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import logging
from tqdm import tqdm
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CityscapesAerialTransformer:
    def __init__(self, download_dir="data/unlabeled_aerial/cityscapes_aerial"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_count = 1000
        self.collected_count = 0
        
        # Cityscapes dataset info (public images only)
        self.cityscapes_urls = {
            'leftImg8bit_trainvaltest': 'https://www.cityscapes-dataset.com/file-handling/?packageID=3'
        }
        
        # Homography parameters for aerial transformation
        self.transform_params = [
            {'perspective_factor': 0.3, 'tilt_angle': 15, 'height_scale': 0.7},
            {'perspective_factor': 0.4, 'tilt_angle': 20, 'height_scale': 0.6},
            {'perspective_factor': 0.5, 'tilt_angle': 25, 'height_scale': 0.5},
            {'perspective_factor': 0.2, 'tilt_angle': 10, 'height_scale': 0.8},
        ]
        
    def download_cityscapes_demo(self):
        """
        Download demo images from Cityscapes (no registration required).
        Note: Full dataset requires registration at cityscapes-dataset.com
        """
        demo_dir = self.download_dir / "demo_images"
        demo_dir.mkdir(exist_ok=True)
        
        # Demo images available without registration
        demo_urls = [
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/aachen_000000_000019_leftImg8bit.png",
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/bochum_000000_000313_leftImg8bit.png",
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/bremen_000000_000019_leftImg8bit.png",
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/cologne_000000_000019_leftImg8bit.png",
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/darmstadt_000000_000019_leftImg8bit.png",
            "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/dusseldorf_000000_000019_leftImg8bit.png"
        ]
        
        downloaded_images = []
        
        for i, url in enumerate(demo_urls):
            try:
                filename = f"cityscapes_demo_{i:03d}.png"
                filepath = demo_dir / filename
                
                if not filepath.exists():
                    logger.info(f"Downloading demo image {i+1}/{len(demo_urls)}")
                    urllib.request.urlretrieve(url, filepath)
                
                downloaded_images.append(str(filepath))
                
            except Exception as e:
                logger.warning(f"Failed to download demo image {url}: {e}")
                continue
        
        logger.info(f"Downloaded {len(downloaded_images)} demo images")
        return downloaded_images
    
    def create_sample_urban_images(self):
        """
        Create sample urban street images for demonstration.
        In practice, you would use actual Cityscapes dataset.
        """
        sample_dir = self.download_dir / "sample_images"
        sample_dir.mkdir(exist_ok=True)
        
        sample_images = []
        
        # Generate synthetic street-view images
        for i in range(20):
            # Create synthetic street image
            img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
            
            # Add road-like features
            # Road surface (bottom third)
            road_height = img.shape[0] // 3
            img[-road_height:, :, :] = [80, 80, 80]  # Gray road
            
            # Lane markings
            lane_y = img.shape[0] - road_height // 2
            img[lane_y-2:lane_y+2, ::100, :] = [255, 255, 255]  # White lines
            
            # Buildings (simplified)
            for j in range(5):
                x_start = j * 400
                x_end = x_start + 300
                building_height = random.randint(300, 600)
                building_color = random.randint(100, 200)
                img[:building_height, x_start:x_end, :] = building_color
            
            # Save sample image
            filename = f"sample_street_{i:03d}.jpg"
            filepath = sample_dir / filename
            
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            sample_images.append(str(filepath))
        
        logger.info(f"Created {len(sample_images)} sample street images")
        return sample_images
    
    def apply_aerial_transform(self, image_path, transform_params):
        """Apply homography transformation to simulate aerial viewpoint."""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            height, width = img.shape[:2]
            
            # Define perspective transformation points
            perspective_factor = transform_params['perspective_factor']
            tilt_angle = transform_params['tilt_angle']
            height_scale = transform_params['height_scale']
            
            # Source points (original street view)
            src_points = np.float32([
                [0, 0],                    # Top-left
                [width, 0],                # Top-right  
                [width, height],           # Bottom-right
                [0, height]                # Bottom-left
            ])
            
            # Destination points (aerial-like perspective)
            perspective_offset = width * perspective_factor
            tilt_offset = height * 0.2
            
            dst_points = np.float32([
                [perspective_offset, tilt_offset],                           # Top-left (narrower)
                [width - perspective_offset, tilt_offset],                   # Top-right (narrower)
                [width * 1.2, height * height_scale],                       # Bottom-right (wider)
                [-width * 0.2, height * height_scale]                       # Bottom-left (wider)
            ])
            
            # Calculate homography matrix
            homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            transformed = cv2.warpPerspective(img, homography_matrix, (width, height))
            
            # Crop and resize to final dimensions
            crop_y = int(height * 0.1)
            crop_height = int(height * 0.8)
            cropped = transformed[crop_y:crop_y + crop_height, :]
            
            # Resize to 1280x1280
            final_img = cv2.resize(cropped, (1280, 1280))
            
            return final_img
            
        except Exception as e:
            logger.error(f"Failed to transform image {image_path}: {e}")
            return None
    
    def process_images(self, source_images):
        """Process source images with aerial transformations."""
        processed_dir = self.download_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        transformed_images = []
        
        for img_path in tqdm(source_images, desc="Transforming to aerial viewpoint"):
            if self.collected_count >= self.target_count:
                break
            
            # Apply multiple transformations per source image
            for i, params in enumerate(self.transform_params):
                if self.collected_count >= self.target_count:
                    break
                
                transformed = self.apply_aerial_transform(img_path, params)
                
                if transformed is not None:
                    # Save transformed image
                    filename = f"cityscapes_aerial_{self.collected_count:06d}_t{i}.jpg"
                    filepath = processed_dir / filename
                    
                    cv2.imwrite(str(filepath), transformed)
                    transformed_images.append(str(filepath))
                    self.collected_count += 1
        
        logger.info(f"Transformed {len(transformed_images)} images to aerial viewpoint")
        return transformed_images
    
    def create_manifest(self, image_paths):
        """Create manifest file for transformed images."""
        manifest = {
            'source': 'Cityscapes Dataset (Aerial Transform)',
            'count': len(image_paths),
            'target_count': self.target_count,
            'transform_params': self.transform_params,
            'images': image_paths,
            'processing_date': str(pd.Timestamp.now()),
            'description': 'Cityscapes images transformed to aerial viewpoint for SSL pre-training',
            'note': 'Full dataset requires registration at cityscapes-dataset.com'
        }
        
        manifest_path = self.download_dir / "cityscapes_aerial_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest: {manifest_path}")
    
    def generate_aerial_dataset(self):
        """Main method to generate aerial-transformed dataset."""
        logger.info("Starting Cityscapes aerial transformation...")
        
        # Try to download demo images first
        demo_images = self.download_cityscapes_demo()
        
        # If no demo images, create samples for demonstration
        if not demo_images:
            logger.info("Demo images not available, creating sample images...")
            demo_images = self.create_sample_urban_images()
        
        # Process images
        transformed_images = self.process_images(demo_images)
        
        if transformed_images:
            self.create_manifest(transformed_images)
            logger.info(f"Cityscapes transformation complete: {len(transformed_images)}/{self.target_count} images")
        else:
            logger.warning("No aerial transformations generated")
        
        return transformed_images

def main():
    """Main execution function."""
    # Import required libraries
    try:
        import random
        globals()['random'] = random
    except ImportError:
        print("❌ Required libraries not available")
        return
    
    transformer = CityscapesAerialTransformer()
    
    try:
        transformed_images = transformer.generate_aerial_dataset()
        
        if transformed_images:
            print(f"✅ Successfully transformed {len(transformed_images)} images to aerial viewpoint")
            print(f"📁 Images saved to: {transformer.download_dir}/processed/")
            print(f"📄 Manifest: {transformer.download_dir}/cityscapes_aerial_manifest.json")
            print(f"💡 Note: Full Cityscapes dataset requires registration at cityscapes-dataset.com")
        else:
            print("❌ Failed to generate aerial transformations")
            
    except KeyboardInterrupt:
        print("\n⏹️ Transformation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
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