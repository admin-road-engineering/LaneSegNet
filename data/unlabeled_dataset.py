#!/usr/bin/env python3
"""
Unlabeled Aerial Dataset for Self-Supervised Learning.
Loads collected unlabeled imagery for MAE pre-training.
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class UnlabeledAerialDataset(Dataset):
    """
    A PyTorch Dataset for loading the unlabeled aerial imagery collected
    by `run_data_collection.bat`. It assumes all images are consolidated
    in a single directory.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the unlabeled images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all image files recursively
        self.image_files = []
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_files.append(os.path.join(root, file))
        
        # Log dataset info
        print(f"Found {len(self.image_files)} unlabeled images in {root_dir}")
        if len(self.image_files) == 0:
            print(f"WARNING: No images found in {root_dir}")
            print("Available subdirectories:")
            if os.path.exists(root_dir):
                for item in os.listdir(root_dir):
                    item_path = os.path.join(root_dir, item)
                    if os.path.isdir(item_path):
                        img_count = len([f for f in os.listdir(item_path) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        print(f"  {item}/: {img_count} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                # Handle both Albumentations and torchvision transforms
                if hasattr(self.transform, '__call__'):
                    # Try Albumentations format first
                    try:
                        image = self.transform(image=np.array(image))['image']
                    except (KeyError, TypeError):
                        # Fallback to torchvision format
                        image = self.transform(image)
                
            return image
            
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                try:
                    fallback = Image.new('RGB', (1280, 1280), color=(0, 0, 0))
                    return self.transform(image=np.array(fallback))['image']
                except:
                    fallback = Image.new('RGB', (1280, 1280), color=(0, 0, 0))
                    return self.transform(fallback)
            else:
                return Image.new('RGB', (224, 224), color=(0, 0, 0))

class ConsolidatedUnlabeledDataset(UnlabeledAerialDataset):
    """
    Specialized dataset that loads from the consolidated directory structure
    created by our data collection pipeline.
    """
    
    def __init__(self, base_dir="data/unlabeled_aerial", transform=None, 
                 prefer_consolidated=True):
        """
        Args:
            base_dir: Base directory containing unlabeled data
            transform: Image transforms
            prefer_consolidated: Whether to prefer consolidated directory
        """
        
        # Try consolidated directory first
        if prefer_consolidated:
            consolidated_dir = os.path.join(base_dir, "consolidated")
            if os.path.exists(consolidated_dir) and len(os.listdir(consolidated_dir)) > 0:
                super().__init__(consolidated_dir, transform)
                print(f"Using consolidated dataset: {consolidated_dir}")
                return
        
        # Fallback: collect from all subdirectories
        print("Consolidated directory not found, collecting from all sources...")
        self.root_dir = base_dir
        self.transform = transform
        self.image_files = []
        
        # Source directories to check
        source_dirs = [
            "osm_1000",
            "osm_test", 
            "skyscapes/processed",
            "carla_synthetic/processed",
            "cityscapes_aerial/processed"
        ]
        
        total_found = 0
        for source_dir in source_dirs:
            full_path = os.path.join(base_dir, source_dir)
            if os.path.exists(full_path):
                source_images = []
                for root, dirs, files in os.walk(full_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            source_images.append(os.path.join(root, file))
                
                self.image_files.extend(source_images)
                total_found += len(source_images)
                print(f"  {source_dir}: {len(source_images)} images")
        
        print(f"Total unlabeled images found: {total_found}")

def test_dataset():
    """Test the unlabeled dataset loading."""
    import torchvision.transforms as transforms
    
    # Test with torchvision transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test consolidated dataset
    dataset = ConsolidatedUnlabeledDataset(transform=transform)
    
    if len(dataset) > 0:
        print(f"Dataset test successful: {len(dataset)} images")
        
        # Test loading first image
        try:
            sample = dataset[0]
            print(f"Sample image shape: {sample.shape}")
            print("Dataset is ready for SSL pre-training!")
        except Exception as e:
            print(f"Error loading sample: {e}")
    else:
        print("No images found - run data collection first")

if __name__ == "__main__":
    test_dataset()