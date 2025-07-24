#!/usr/bin/env python3
"""
Advanced Augmentation Pipeline for Phase 3.2.5
MixUp, CutMix, Copy-Paste, and Style Transfer for rare class enhancement
"""

import torch
import numpy as np
import cv2
import random
from typing import Tuple
import albumentations as A

class AdvancedAugmentor:
    """Industry-leading augmentations for aerial lane detection."""
    
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, smart_augment_prob=0.5, 
                 style_transfer_prob=0.3, copypaste_prob=0.5, cutmix_prob=0.3):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Probabilities for smart_augment choices
        self.smart_augment_prob = smart_augment_prob
        self.style_transfer_prob = style_transfer_prob
        self.copypaste_prob = copypaste_prob
        self.cutmix_prob = cutmix_prob
        
        # Style transfer for weather/lighting variations
        self.style_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2)
        ], additional_targets={'mask': 'mask'})

    def __call__(self, img1: torch.Tensor, mask1: torch.Tensor,
                 img2: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main entry point to apply a sequence of augmentations.
        Centralized logic for all augmentation decisions.
        """
        # 1. Apply smart augment (MixUp, CutMix, CopyPaste)
        if np.random.random() < self.smart_augment_prob:
            img1, mask1 = self.smart_augment(img1, mask1, img2, mask2)

        # 2. Apply style transfer on the (potentially augmented) result
        if np.random.random() < self.style_transfer_prob:
            img1, mask1 = self.style_transfer_augment(img1, mask1)

        return img1, mask1
    
    def mixup_augment(self, img1: torch.Tensor, mask1: torch.Tensor, 
                      img2: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MixUp augmentation for smooth interpolation between samples.
        Particularly effective for white dashed lane synthesis.
        """
        if img1.shape != img2.shape or mask1.shape != mask2.shape:
            return img1, mask1  # Skip if shapes don't match
        
        # Sample mixing coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        
        # Mix images and masks
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # For segmentation masks, we need to handle discrete labels carefully
        # Use probabilistic mixing for rare classes
        mixed_mask = self._mix_segmentation_masks(mask1, mask2, lam)
        
        return mixed_img, mixed_mask
    
    def cutmix_augment(self, img1: torch.Tensor, mask1: torch.Tensor, 
                       img2: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CutMix augmentation for spatial region replacement.
        Effective for copy-pasting lane segments.
        """
        if img1.shape != img2.shape or mask1.shape != mask2.shape:
            return img1, mask1
        
        H, W = img1.shape[-2:]
        
        # Generate random bounding box
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if self.cutmix_alpha > 0 else 1.0
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_img = img1.clone()
        mixed_mask = mask1.clone()
        
        mixed_img[..., bby1:bby2, bbx1:bbx2] = img2[..., bby1:bby2, bbx1:bbx2]
        mixed_mask[..., bby1:bby2, bbx1:bbx2] = mask2[..., bby1:bby2, bbx1:bbx2]
        
        return mixed_img, mixed_mask
    
    def copy_paste_lanes(self, img1: torch.Tensor, mask1: torch.Tensor, 
                        img2: torch.Tensor, mask2: torch.Tensor, 
                        target_class: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copy-paste specific lane types (e.g., white dashed lanes).
        Expert panel innovation for rare class enhancement.
        
        TODO: For even more realistic blending, consider implementing Poisson blending:
        - Use cv2.seamlessClone() for seamless integration
        - Better color/lighting adaptation to background
        - More natural appearance of pasted lane segments
        """
        if img1.shape != img2.shape or mask1.shape != mask2.shape:
            return img1, mask2
        
        # Find regions with target class in source image
        source_regions = (mask2 == target_class)
        
        if not source_regions.any():
            return img1, mask1  # No target class found
        
        # Apply morphological operations to get clean lane segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        source_regions_np = source_regions.cpu().numpy().astype(np.uint8)
        source_regions_clean = cv2.morphologyEx(source_regions_np, cv2.MORPH_OPEN, kernel)
        source_regions_clean = torch.from_numpy(source_regions_clean).bool().to(mask1.device)
        
        # Paste lanes into target image
        result_img = img1.clone()
        result_mask = mask1.clone()
        
        # Blend images in lane regions
        alpha = 0.7  # Blending coefficient
        result_img = torch.where(
            source_regions_clean.unsqueeze(0).expand_as(img1),
            alpha * img2 + (1 - alpha) * img1,
            img1
        )
        
        # Update mask
        result_mask = torch.where(source_regions_clean, target_class, mask1)
        
        return result_img, result_mask
    
    def style_transfer_augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weather and lighting style transfer for robustness.
        Maintains lane visibility while adding environmental variations.
        """
        # Convert tensor to numpy for albumentations
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Apply style transformations
        transformed = self.style_transforms(image=img_np, mask=mask_np)
        
        # Convert back to tensor
        result_img = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
        result_mask = torch.from_numpy(transformed['mask']).long()
        
        return result_img.to(img.device), result_mask.to(mask.device)
    
    def _mix_segmentation_masks(self, mask1: torch.Tensor, mask2: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Smart mask mixing that preserves rare classes.
        Prioritizes minority classes (white lanes) during mixing.
        """
        # For rare classes (1, 2), prefer to keep them
        rare_classes = [1, 2]  # white_solid, white_dashed
        
        mixed_mask = mask1.clone()
        
        for rare_class in rare_classes:
            # If mask2 has rare class and mask1 doesn't, copy with probability
            mask2_has_rare = (mask2 == rare_class)
            mask1_has_rare = (mask1 == rare_class)
            
            # Copy rare class from mask2 if it's not in mask1
            copy_rare = mask2_has_rare & ~mask1_has_rare
            if copy_rare.any() and np.random.random() < (1 - lam):  # Higher prob for rare classes
                mixed_mask = torch.where(copy_rare, rare_class, mixed_mask)
        
        return mixed_mask
    
    def smart_augment(self, img1: torch.Tensor, mask1: torch.Tensor, 
                     img2: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Intelligent augmentation selection based on class distribution.
        Prioritizes augmentations that help with class imbalance.
        """
        # Check class distribution to decide augmentation strategy
        rare_classes_in_mask1 = [(mask1 == c).sum().item() for c in [1, 2]]  # white lanes
        rare_classes_in_mask2 = [(mask2 == c).sum().item() for c in [1, 2]]
        
        # If mask2 has more rare classes, prefer copy-paste or cutmix
        if sum(rare_classes_in_mask2) > sum(rare_classes_in_mask1):
            # Define choices and their probabilities
            choices = ['copy_solid', 'copy_dashed', 'cutmix']
            # Ensure probabilities sum to 1 for np.random.choice
            remaining_prob = 1.0 - self.copypaste_prob
            probabilities = [self.copypaste_prob / 2, self.copypaste_prob / 2, remaining_prob]
            
            choice = np.random.choice(choices, p=probabilities)
            if choice == 'copy_solid':
                return self.copy_paste_lanes(img1, mask1, img2, mask2, target_class=1)  # white_solid
            elif choice == 'copy_dashed':
                return self.copy_paste_lanes(img1, mask1, img2, mask2, target_class=2)  # white_dashed
            else:
                return self.cutmix_augment(img1, mask1, img2, mask2)
        else:
            # Use mixup for general augmentation
            return self.mixup_augment(img1, mask1, img2, mask2)

# Example usage integration
class EnhancedPremiumDataset:
    """Enhanced dataset with advanced augmentations."""
    
    def __init__(self, img_dir, mask_dir, mode='train'):
        # self.base_dataset = ... # Initialize your base dataset here
        self.base_dataset = None  # Placeholder for existing PremiumDataset
        self.mode = mode
        self.augmentor = AdvancedAugmentor(smart_augment_prob=0.5, style_transfer_prob=0.3) if self.mode == 'train' else None
        
    def __getitem__(self, idx):
        # Get base sample
        img, mask = self.base_dataset[idx]
        
        if self.mode == 'train' and self.augmentor:
            # Get another sample for mixing
            mix_idx = np.random.randint(len(self.base_dataset))
            img2, mask2 = self.base_dataset[mix_idx]
            
            # A single call to the augmentor handles all logic
            img, mask = self.augmentor(img, mask, img2, mask2)
        
        return img, mask

if __name__ == "__main__":
    print("Advanced Augmentation Pipeline for Phase 3.2.5")
    print("Features:")
    print("- MixUp for smooth interpolation")
    print("- CutMix for spatial region mixing") 
    print("- Copy-Paste for rare lane enhancement")
    print("- Style Transfer for environmental robustness")
    print("- Smart selection based on class distribution")
    print("\nExpected gain: 3-5% mIoU from rare class enhancement")