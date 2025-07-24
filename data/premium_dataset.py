#!/usr/bin/env python3
"""
Labeled Aerial Dataset for Supervised Fine-Tuning.
Loads the premium AEL dataset with advanced augmentations.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PremiumDataset(Dataset):
    """
    Premium dataset with advanced augmentations for maximum quality.
    This class is used for loading the labeled training and validation data.
    """
    def __init__(self, img_dir, mask_dir, mode='train', img_size=(512, 512)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.images = sorted(list(self.img_dir.glob("*.jpg")))
        self.img_size = img_size

        # Advanced augmentation pipeline for training
        if mode == 'train':
            self.color_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
            ])

        print(f"Premium Labeled Dataset '{mode}': {len(self.images)} samples found in {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        mask = np.clip(mask, 0, 3) # Assuming max 4 classes (0, 1, 2, 3)

        # Advanced augmentations for training
        if self.mode == 'train':
            # Geometric augmentations
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            # Rotation (small angles to preserve lane structure)
            if np.random.random() > 0.7:
                angle = np.random.uniform(-10, 10)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Scale variation
            if np.random.random() > 0.8:
                scale = np.random.uniform(0.9, 1.1)
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Crop or pad to maintain size
                if scale > 1.0:
                    start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
                    image = image[start_h:start_h + h, start_w:start_w + w]
                    mask = mask[start_h:start_h + h, start_w:start_w + w]
                else:
                    pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                    image = cv2.copyMakeBorder(image, pad_h, h - new_h - pad_h,
                                             pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
                    mask = cv2.copyMakeBorder(mask, pad_h, h - new_h - pad_h,
                                            pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT, value=0)

        # Convert to tensors
        image = torch.from_numpy(image.copy().transpose(2, 0, 1)).float() / 255.0

        # Apply color augmentations (preserves lane visibility)
        if self.mode == 'train' and np.random.random() > 0.6:
            image = self.color_transforms(image)

        mask = torch.from_numpy(mask.copy()).long()

        return image, mask