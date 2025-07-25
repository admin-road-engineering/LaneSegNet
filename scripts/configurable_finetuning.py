#!/usr/bin/env python3
"""
Configurable Fine-Tuning Script for Hyperparameter Sweeping.
Refactored from run_finetuning.py to expose all hyperparameters via command-line arguments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import json
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np

# Add transformers for warmup scheduler
try:
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    HF_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available - using PyTorch schedulers only")
    HF_AVAILABLE = False

# Add albumentations for advanced augmentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    logger.warning("albumentations not available - using basic augmentations only")
    ALBUMENTATIONS_AVAILABLE = False

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our components (same as original)
from scripts.ssl_pretraining import MaskedAutoencoderViT
from scripts.ohem_loss import OHEMDiceFocalLoss, AdaptiveOHEMLoss
from scripts.enhanced_post_processing import EnhancedPostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interpolate_pos_embed(pos_embed_checkpoint, new_img_size, patch_size):
    """Same as original - interpolates positional embeddings."""
    embedding_dim = pos_embed_checkpoint.shape[-1]
    num_patches_old = pos_embed_checkpoint.shape[1]
    old_grid_size = int(num_patches_old**0.5)
    new_grid_size = new_img_size // patch_size
    
    logger.info(f"Interpolating pos_embed from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")
    
    pos_embed_grid = pos_embed_checkpoint.transpose(1, 2).reshape(1, embedding_dim, old_grid_size, old_grid_size)
    pos_embed_interpolated = F.interpolate(
        pos_embed_grid, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False
    )
    
    return pos_embed_interpolated.flatten(2).transpose(1, 2)

class PretrainedLaneNet(nn.Module):
    """Same as original model architecture - no changes needed."""
    
    def __init__(self, num_classes=NUM_CLASSES, img_size=512, encoder_weights_path=None, freeze_encoder=False):
        super().__init__()
        
        # Create MAE model to extract encoder
        mae_model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            mask_ratio=0.75
        )
        
        # Load pre-trained weights if provided
        if encoder_weights_path and os.path.exists(encoder_weights_path):
            logger.info(f"Loading pre-trained encoder from: {encoder_weights_path}")
            checkpoint = torch.load(encoder_weights_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            if 'pos_embed' in state_dict and state_dict['pos_embed'].shape != mae_model.pos_embed.shape:
                pos_embed_checkpoint = state_dict['pos_embed']
                pos_embed_interpolated = interpolate_pos_embed(pos_embed_checkpoint, img_size, 16)
                state_dict['pos_embed'] = pos_embed_interpolated
                logger.info("Successfully interpolated positional embeddings.")
            
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(('patch_embed', 'encoder', 'pos_embed')):
                    encoder_state_dict[key] = value
            
            mae_model.load_state_dict(encoder_state_dict, strict=False)
            logger.info("Pre-trained encoder weights loaded successfully.")
        else:
            logger.warning("No pre-trained weights provided - using random initialization")
        
        # Extract encoder components
        self.patch_embed = mae_model.patch_embed
        self.pos_embed = mae_model.pos_embed
        self.encoder = mae_model.encoder
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in [self.patch_embed.parameters(), self.encoder.parameters()]:
                for p in param:
                    p.requires_grad = False
            self.pos_embed.requires_grad = False
            logger.info("Encoder frozen - only decoder will be trained")
        
        # Task-specific decoder for lane segmentation
        self.decoder = self._build_segmentation_decoder(
            embed_dim=768,
            num_classes=num_classes,
            img_size=img_size
        )
        
        self._log_model_info()
    
    def _build_segmentation_decoder(self, embed_dim, num_classes, img_size):
        """Build decoder for semantic segmentation."""
        decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Upsampling layers to restore full resolution
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Final classification layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        return decoder
    
    def _log_model_info(self):
        """Log model parameter information."""
        encoder_params = sum(p.numel() for p in self.patch_embed.parameters())
        encoder_params += sum(p.numel() for p in self.encoder.parameters())
        encoder_params += self.pos_embed.numel()
        
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        decoder_params += sum(p.numel() for p in self.upsample_layers.parameters())
        decoder_params += sum(p.numel() for p in self.final_conv.parameters())
        
        total_params = encoder_params + decoder_params
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"PretrainedLaneNet created:")
        logger.info(f"  Encoder parameters: {encoder_params:,}")
        logger.info(f"  Decoder parameters: {decoder_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def forward(self, x):
        """Forward pass through the model."""
        B, C, H, W = x.shape
        
        # Encoder: Image -> Patch embeddings -> Transformer features
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        
        # Apply decoder transformation
        x = self.decoder(x)
        
        # Reshape back to spatial format
        patch_size = 16
        num_patches_per_side = H // patch_size
        x = x.transpose(1, 2).reshape(B, -1, num_patches_per_side, num_patches_per_side)
        
        # Progressive upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

class ConfigurableFineTuningTrainer:
    """
    REFACTORED: All hyperparameters now configurable via constructor arguments.
    """
    
    def __init__(self, model, device, save_dir="work_dirs/finetuning",
                 # Optimizer hyperparameters (EXPOSED)
                 encoder_lr=1e-5, decoder_lr=5e-4, weight_decay=1e-4,
                 optimizer_type='adamw', beta1=0.9, beta2=0.999,
                 # Scheduler hyperparameters (EXPOSED)
                 scheduler_type='plateau', scheduler_factor=0.5, 
                 scheduler_patience=5, scheduler_threshold=1e-4,
                 warmup_steps=100, total_steps=None,
                 # Loss function hyperparameters (EXPOSED)
                 use_ohem=True, ohem_alpha=0.25, ohem_gamma=2.0, 
                 ohem_thresh=0.7, ohem_min_kept=512,
                 dice_weight=0.6, focal_weight=0.4,
                 # Training hyperparameters (EXPOSED)
                 early_stopping_patience=10, gradient_clip_norm=None,
                 # Post-processing (EXPOSED)
                 use_enhanced_postprocessing=False):
        
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_enhanced_postprocessing = use_enhanced_postprocessing
        
        # Early stopping parameters (CONFIGURABLE)
        self.best_iou = 0.0
        self.patience = early_stopping_patience
        self.early_stop_counter = 0
        self.gradient_clip_norm = gradient_clip_norm
        
        # Loss function configuration (CONFIGURABLE)
        if use_ohem:
            self.criterion = OHEMDiceFocalLoss(
                alpha=ohem_alpha,
                gamma=ohem_gamma,
                smooth=1e-6,
                ohem_thresh=ohem_thresh,
                ohem_min_kept=ohem_min_kept,
                dice_weight=dice_weight,
                focal_weight=focal_weight
            )
            logger.info(f"Using OHEM DiceFocal loss (alpha={ohem_alpha}, gamma={ohem_gamma}, thresh={ohem_thresh})")
        else:
            from scripts.premium_gpu_train import DiceFocalLoss
            self.criterion = DiceFocalLoss(alpha=ohem_alpha, gamma=ohem_gamma, smooth=1e-6)
            logger.info("Using standard DiceFocal loss")
        
        # Optimizer configuration (CONFIGURABLE)
        encoder_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if name.startswith(('patch_embed', 'pos_embed', 'encoder')):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Create optimizer with configurable parameters
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': encoder_lr},
                {'params': decoder_params, 'lr': decoder_lr}
            ], weight_decay=weight_decay, betas=(beta1, beta2))
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam([
                {'params': encoder_params, 'lr': encoder_lr},
                {'params': decoder_params, 'lr': decoder_lr}
            ], weight_decay=weight_decay, betas=(beta1, beta2))
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD([
                {'params': encoder_params, 'lr': encoder_lr},
                {'params': decoder_params, 'lr': decoder_lr}
            ], weight_decay=weight_decay, momentum=beta1)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Learning rate scheduler configuration (CONFIGURABLE)
        self.scheduler_type = scheduler_type.lower()
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        if self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=scheduler_factor, 
                patience=scheduler_patience, threshold=scheduler_threshold, verbose=True
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scheduler_patience, gamma=scheduler_factor
            )
        elif self.scheduler_type == 'none':
            self.scheduler = None
        elif self.scheduler_type in ['cosine', 'cosine-warmup', 'linear-warmup']:
            # Will be configured with total steps in train() method
            self.scheduler = None  # Initialize later
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Enhanced post-processing (CONFIGURABLE)
        if use_enhanced_postprocessing:
            self.post_processor = EnhancedPostProcessor(
                use_tta=True, use_morphology=True, use_crf=True
            )
            logger.info("Enhanced post-processing enabled")
        
        self.training_log = []
        
        # Log all configuration
        logger.info("Configurable fine-tuning trainer initialized:")
        logger.info(f"  Encoder LR: {encoder_lr}")
        logger.info(f"  Decoder LR: {decoder_lr}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Optimizer: {optimizer_type}")
        logger.info(f"  Scheduler: {scheduler_type}")
        logger.info(f"  Early stopping patience: {early_stopping_patience}")
        logger.info(f"  Gradient clipping: {gradient_clip_norm}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch - same as original."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (CONFIGURABLE)
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                'enc_lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'dec_lr': f"{self.optimizer.param_groups[1]['lr']:.6f}"
            })
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader):
        """Evaluate model - same as original."""
        self.model.eval()
        total_iou = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_enhanced_postprocessing:
                    predictions = self.post_processor.process(self.model, images)
                else:
                    outputs = self.model(images)
                    predictions = torch.argmax(outputs, dim=1)
                
                # Calculate IoU for each sample
                for pred, target in zip(predictions, targets):
                    iou = self._calculate_iou(pred, target)
                    total_iou += iou
                    num_samples += 1
        
        return total_iou / num_samples if num_samples > 0 else 0.0
    
    def _calculate_iou(self, pred, target):
        """Calculate mean IoU - same as original with dynamic class detection."""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        num_classes = self.model.final_conv.out_channels
        ious = []
        
        for class_id in range(1, num_classes):  # Skip background
            pred_mask = (pred_np == class_id)
            target_mask = (target_np == class_id)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        
        return np.nanmean(ious) if ious else 0.0
    
    def save_checkpoint(self, epoch, loss, iou, is_best=False):
        """Save model checkpoint - same as original."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'iou': iou,
            'training_log': self.training_log
        }
        
        checkpoint_path = self.save_dir / f"finetuned_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / "finetuned_best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path} (IoU: {iou:.1%})")
        
        return checkpoint_path
    
    def train(self, train_loader, val_loader, epochs=100):
        """Full training loop with configurable scheduler initialization."""
        # Calculate total steps if not provided
        if self.total_steps is None:
            self.total_steps = epochs * len(train_loader)
        
        # Initialize warmup and advanced schedulers
        if self.scheduler is None and self.scheduler_type in ['cosine', 'cosine-warmup', 'linear-warmup']:
            if self.scheduler_type == 'cosine':
                T_max = int(epochs * self.scheduler_factor) if self.scheduler_factor != 0.5 else epochs
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max, eta_min=1e-7
                )
                logger.info(f"Cosine annealing scheduler initialized with T_max={T_max}")
            elif self.scheduler_type == 'cosine-warmup' and HF_AVAILABLE:
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.total_steps,
                    num_cycles=0.5
                )
                logger.info(f"Cosine warmup scheduler initialized with {self.warmup_steps} warmup steps")
            elif self.scheduler_type == 'linear-warmup' and HF_AVAILABLE:
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.total_steps
                )
                logger.info(f"Linear warmup scheduler initialized with {self.warmup_steps} warmup steps")
            elif not HF_AVAILABLE:
                logger.warning(f"Transformers not available - falling back to cosine annealing")
                T_max = int(epochs * self.scheduler_factor) if self.scheduler_factor != 0.5 else epochs
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max, eta_min=1e-7
                )
        
        logger.info(f"Starting fine-tuning for {epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Total training steps: {self.total_steps}")
        
        for epoch in range(epochs):
            # Training
            avg_loss = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            avg_iou = self.evaluate(val_loader)
            
            # Update training log
            self.training_log.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'iou': avg_iou,
                'encoder_lr': self.optimizer.param_groups[0]['lr'],
                'decoder_lr': self.optimizer.param_groups[1]['lr']
            })
            
            # Check if best model and update early stopping
            is_best = avg_iou > self.best_iou
            if is_best:
                self.best_iou = avg_iou
                self.early_stop_counter = 0
                best_model_path = self.save_dir / "best_model.pth"
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"New best model saved: {best_model_path}")
            else:
                self.early_stop_counter += 1
                
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_iou)
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.1%} "
                f"{'(NEW BEST!)' if is_best else ''}"
            )
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, avg_loss, avg_iou, is_best)
                
            # Early stopping check
            if self.early_stop_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience: {self.patience})")
                logger.info(f"Best IoU achieved: {self.best_iou:.1%}")
                break
        
        # Save final results
        self.save_checkpoint(epochs, avg_loss, avg_iou, is_best=False)
        
        log_path = self.save_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        logger.info("Fine-tuning completed!")
        logger.info(f"Best IoU achieved: {self.best_iou:.1%}")
        logger.info(f"Models saved to: {self.save_dir}")

def create_augmentation_transforms(augmentation_level='basic'):
    """Create augmentation transforms based on level."""
    if not ALBUMENTATIONS_AVAILABLE:
        logger.warning("Albumentations not available - returning None transforms")
        return None, None
    
    # Validation transform (same for all levels)
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    if augmentation_level == 'none':
        train_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif augmentation_level == 'basic':
        train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    elif augmentation_level == 'strong':
        train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Simulates occlusions
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    logger.info(f"Created '{augmentation_level}' augmentation transforms")
    return train_transform, val_transform

class AugmentedDatasetWrapper:
    """Wrapper to apply Albumentations transforms to existing dataset."""
    
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # CHW -> HWC
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is long tensor for loss computation
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        elif mask.dtype != torch.long:
            mask = mask.long()
        
        return image, mask

def create_dataloaders(data_dir="data/ael_mmseg", img_size=512, batch_size=4, num_workers=4, augmentation_level='basic'):
    """Create dataloaders with configurable augmentations."""
    try:
        from data.labeled_dataset import LabeledDataset
        
        img_size_tuple = (img_size, img_size)
        
        # Create base datasets
        train_base_dataset = LabeledDataset(
            os.path.join(data_dir, "img_dir/train"),
            os.path.join(data_dir, "ann_dir/train"),
            mode='train', img_size=img_size_tuple
        )
        
        val_base_dataset = LabeledDataset(
            os.path.join(data_dir, "img_dir/val"),
            os.path.join(data_dir, "ann_dir/val"),
            mode='val', img_size=img_size_tuple
        )
        
        # Create augmentation transforms
        train_transform, val_transform = create_augmentation_transforms(augmentation_level)
        
        # Wrap datasets with augmentations
        if ALBUMENTATIONS_AVAILABLE and train_transform is not None:
            train_dataset = AugmentedDatasetWrapper(train_base_dataset, train_transform)
            val_dataset = AugmentedDatasetWrapper(val_base_dataset, val_transform)
            logger.info(f"Applied '{augmentation_level}' augmentations to datasets")
        else:
            train_dataset = train_base_dataset
            val_dataset = val_base_dataset
            logger.warning("Using base datasets without augmentations")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader
        
    except ImportError as e:
        logger.error(f"Could not import LabeledDataset: {e}")
        raise

def main():
    """Main function with EXPANDED command-line arguments for hyperparameter sweeping."""
    parser = argparse.ArgumentParser(description="Configurable fine-tuning for hyperparameter sweeping")
    
    # Model parameters (same as original)
    parser.add_argument('--encoder-weights', type=str, 
                       default='work_dirs/mae_pretraining/mae_best_model.pth',
                       help='Path to pre-trained encoder weights')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder during fine-tuning')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of output classes')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size for fine-tuning')
    
    # Training parameters (same as original)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='data/ael_mmseg',
                       help='Data directory with train/val splits')
    
    # EXPANDED: Optimizer hyperparameters
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                       help='Learning rate for encoder (pre-trained) parameters')
    parser.add_argument('--decoder-lr', type=float, default=5e-4,
                       help='Learning rate for decoder (new) parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Beta1 for Adam/AdamW or momentum for SGD')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam/AdamW')
    
    # EXPANDED: Scheduler hyperparameters
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'cosine-warmup', 'linear-warmup', 'step', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                       help='Factor for scheduler (decay factor or T_max multiplier)')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                       help='Patience for plateau scheduler or step size for step scheduler')
    parser.add_argument('--scheduler-threshold', type=float, default=1e-4,
                       help='Threshold for plateau scheduler')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps for warmup schedulers')
    parser.add_argument('--total-steps', type=int, default=None,
                       help='Total training steps (overrides epochs calculation if set)')
    
    # EXPANDED: Loss function hyperparameters
    parser.add_argument('--use-ohem', action='store_true', default=True,
                       help='Use OHEM loss function')
    parser.add_argument('--ohem-alpha', type=float, default=0.25,
                       help='Alpha parameter for focal loss component')
    parser.add_argument('--ohem-gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss component')
    parser.add_argument('--ohem-thresh', type=float, default=0.7,
                       help='OHEM threshold for hard example mining')
    parser.add_argument('--ohem-min-kept', type=int, default=512,
                       help='Minimum number of examples to keep in OHEM')
    parser.add_argument('--dice-weight', type=float, default=0.6,
                       help='Weight for dice loss component')
    parser.add_argument('--focal-weight', type=float, default=0.4,
                       help='Weight for focal loss component')
    
    # EXPANDED: Training hyperparameters
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--gradient-clip-norm', type=float, default=None,
                       help='Gradient clipping norm (None to disable)')
    parser.add_argument('--augmentation-level', type=str, default='basic',
                       choices=['none', 'basic', 'strong'],
                       help='Data augmentation level for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Advanced techniques (same as original)
    parser.add_argument('--use-enhanced-postprocessing', action='store_true',
                       help='Use enhanced post-processing during evaluation')
    
    # System parameters (same as original)
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='work_dirs/configurable_finetuning',
                       help='Directory to save results')
    
    # NEW: Experiment naming for sweeps
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (will be appended to save-dir)')
    
    args = parser.parse_args()
    
    # Create experiment-specific save directory
    if args.experiment_name:
        args.save_dir = os.path.join(args.save_dir, args.experiment_name)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("=" * 100)
    print("CONFIGURABLE FINE-TUNING FOR HYPERPARAMETER SWEEPING")
    print("=" * 100)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Experiment: {args.experiment_name or 'default'}")
    print(f"Save directory: {args.save_dir}")
    print(f"Seed: {args.seed}")
    print(f"Pre-trained weights: {args.encoder_weights}")
    print(f"Optimizer: {args.optimizer} (enc_lr={args.encoder_lr}, dec_lr={args.decoder_lr}, wd={args.weight_decay})")
    print(f"Scheduler: {args.scheduler} (factor={args.scheduler_factor}, warmup={args.warmup_steps})")
    print(f"Loss: OHEM={args.use_ohem} (alpha={args.ohem_alpha}, gamma={args.ohem_gamma})")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, patience={args.early_stopping_patience}")
    print(f"Augmentations: {args.augmentation_level}")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model
        model = PretrainedLaneNet(
            num_classes=args.num_classes,
            img_size=args.img_size,
            encoder_weights_path=args.encoder_weights,
            freeze_encoder=args.freeze_encoder
        )
        
        # Create dataloaders with augmentations
        train_loader, val_loader = create_dataloaders(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentation_level=args.augmentation_level
        )
        
        # Create configurable trainer
        trainer = ConfigurableFineTuningTrainer(
            model=model,
            device=device,
            save_dir=args.save_dir,
            # Pass all hyperparameters to trainer
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            beta1=args.beta1,
            beta2=args.beta2,
            scheduler_type=args.scheduler,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            scheduler_threshold=args.scheduler_threshold,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            use_ohem=args.use_ohem,
            ohem_alpha=args.ohem_alpha,
            ohem_gamma=args.ohem_gamma,
            ohem_thresh=args.ohem_thresh,
            ohem_min_kept=args.ohem_min_kept,
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_norm=args.gradient_clip_norm,
            use_enhanced_postprocessing=args.use_enhanced_postprocessing
        )
        
        # Start training
        trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        print("\n" + "=" * 100)
        print("CONFIGURABLE FINE-TUNING COMPLETED!")
        print("=" * 100)
        print(f"Best IoU achieved: {trainer.best_iou:.1%}")
        print(f"Final model saved to: {args.save_dir}/finetuned_best_model.pth")
        print("=" * 100)
        
        # Save hyperparameter configuration for sweep analysis
        config_path = Path(args.save_dir) / "hyperparameter_config.json"
        config = {k: v for k, v in vars(args).items()}
        config['best_iou'] = trainer.best_iou
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return trainer.best_iou  # Return best IoU for sweep analysis
        
    except Exception as e:
        logger.error(f"Configurable fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    best_iou = main()
    # Exit with success code based on performance (can be used by sweep framework)
    sys.exit(0 if best_iou > 0 else 1)