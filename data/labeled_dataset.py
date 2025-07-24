#!/usr/bin/env python3
"""
Standardized Labeled Dataset for Supervised Fine-Tuning.
This is the single source of truth for loading labeled data.
It uses the complete 7,817 sample dataset and robust augmentations.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LabeledDataset(Dataset):
    """
    Standardized dataset class for loading labeled training, validation,
    and test data. Uses albumentations for robust data augmentation.
    """
    def __init__(self, img_dir, ann_dir, mode='train', img_size=(512, 512)):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.mode = mode
        
        # Get all image files and verify matching annotations
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
        self.valid_files = [
            img_file.stem for img_file in self.image_files 
            if (self.ann_dir / f"{img_file.stem}.png").exists()
        ]
        
        print(f"Labeled Dataset '{mode}': {len(self.valid_files)} valid samples found in {img_dir}")
        
        # Define augmentations using Albumentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                ], p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else: # For validation and testing
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        sample_id = self.valid_files[idx]
        
        # Load image
        img_path = self.img_dir / f"{sample_id}.jpg"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.ann_dir / f"{sample_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        
        return transformed['image'], transformed['mask'].long()