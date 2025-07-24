#!/usr/bin/env python3
"""
Download SkyScapes dataset for unlabeled aerial imagery.
Target: ~3k aerial road images for self-supervised pre-training.
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkyScapesDownloader:
    def __init__(self, download_dir="data/unlabeled_aerial/skyscapes"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # SkyScapes dataset URLs (public dataset)
        self.dataset_urls = {
            'dense_images': 'https://www.skyscapes.ml/files/skyscapes_dense_images.zip',
            'sparse_images': 'https://www.skyscapes.ml/files/skyscapes_sparse_images.zip'
        }
        
        self.target_count = 3000
        self.collected_count = 0
        
    def download_file(self, url, filename):
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=f"Downloading {filename.name}",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {filename}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Extracted: {zip_path}")
            return True
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
    
    def is_aerial_viewpoint(self, image_path):
        """
        Heuristic to determine if image has aerial/bird's eye viewpoint.
        SkyScapes has mixed viewpoints - we want aerial-like images.
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Check image dimensions (aerial images often wider than tall)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Aerial images typically have:
            # 1. Wider aspect ratio (>1.2)
            # 2. More horizontal structures
            # 3. Less sky (top pixels not predominantly blue/white)
            
            if aspect_ratio < 1.2:
                return False
            
            # Check top 20% of image for sky content
            top_section = img_array[:int(height * 0.2), :]
            
            # Convert to HSV for better sky detection
            if len(img_array.shape) == 3:
                # Simple blue/white detection in RGB
                blue_mask = (top_section[:,:,2] > top_section[:,:,0]) & (top_section[:,:,2] > top_section[:,:,1])
                white_mask = np.all(top_section > 200, axis=2)
                sky_ratio = (np.sum(blue_mask) + np.sum(white_mask)) / top_section[:,:,0].size
                
                # If top 20% is mostly sky (>60%), likely not aerial view
                if sky_ratio > 0.6:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_images(self, source_dir):
        """Process images and select aerial viewpoints."""
        processed_dir = self.download_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        image_files = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png"))
        logger.info(f"Found {len(image_files)} images to process")
        
        selected_images = []
        
        for img_path in tqdm(image_files, desc="Filtering aerial viewpoints"):
            if self.collected_count >= self.target_count:
                break
                
            if self.is_aerial_viewpoint(img_path):
                # Copy to processed directory
                new_name = f"skyscapes_aerial_{self.collected_count:06d}.jpg"
                new_path = processed_dir / new_name
                
                try:
                    # Resize to consistent dimensions (1280x1280 to match our training data)
                    img = Image.open(img_path)
                    img_resized = img.resize((1280, 1280), Image.LANCZOS)
                    img_resized.save(new_path, "JPEG", quality=90)
                    
                    selected_images.append(str(new_path))
                    self.collected_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
        
        logger.info(f"Selected {len(selected_images)} aerial images from SkyScapes")
        return selected_images
    
    def create_manifest(self, image_paths):
        """Create manifest file for selected images."""
        manifest = {
            'source': 'SkyScapes Dataset',
            'count': len(image_paths),
            'target_count': self.target_count,
            'images': image_paths,
            'processing_date': str(pd.Timestamp.now()),
            'description': 'Aerial viewpoint images filtered from SkyScapes dataset for SSL pre-training'
        }
        
        manifest_path = self.download_dir / "skyscapes_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest: {manifest_path}")
    
    def download_and_process(self):
        """Main method to download and process SkyScapes data."""
        logger.info("Starting SkyScapes dataset download...")
        
        # Create temporary download directory
        temp_dir = self.download_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        selected_images = []
        
        for dataset_name, url in self.dataset_urls.items():
            if self.collected_count >= self.target_count:
                break
                
            logger.info(f"Processing {dataset_name}...")
            
            # Download zip file
            zip_path = temp_dir / f"{dataset_name}.zip"
            if not zip_path.exists():
                if not self.download_file(url, zip_path):
                    logger.warning(f"Skipping {dataset_name} due to download failure")
                    continue
            
            # Extract zip file
            extract_dir = temp_dir / dataset_name
            if not extract_dir.exists():
                if not self.extract_zip(zip_path, extract_dir):
                    logger.warning(f"Skipping {dataset_name} due to extraction failure")
                    continue
            
            # Process images
            batch_images = self.process_images(extract_dir)
            selected_images.extend(batch_images)
            
            logger.info(f"Collected {len(batch_images)} from {dataset_name}")
        
        # Create manifest
        if selected_images:
            self.create_manifest(selected_images)
            logger.info(f"SkyScapes download complete: {len(selected_images)}/{self.target_count} images")
        else:
            logger.warning("No aerial images collected from SkyScapes")
        
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
        
        return selected_images

def main():
    """Main execution function."""
    downloader = SkyScapesDownloader()
    
    try:
        selected_images = downloader.download_and_process()
        
        if selected_images:
            print(f"✅ Successfully collected {len(selected_images)} aerial images from SkyScapes")
            print(f"📁 Images saved to: {downloader.download_dir}/processed/")
            print(f"📄 Manifest: {downloader.download_dir}/skyscapes_manifest.json")
        else:
            print("❌ Failed to collect aerial images from SkyScapes")
            
    except KeyboardInterrupt:
        print("\n⏹️ Download interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    # Add pandas import for timestamp
    try:
        import pandas as pd
    except ImportError:
        # Fallback to datetime if pandas not available
        import datetime as dt
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return dt.datetime.now().isoformat()
    
    main()