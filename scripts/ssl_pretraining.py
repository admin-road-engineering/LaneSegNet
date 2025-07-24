#!/usr/bin/env python3
"""
Self-Supervised Pre-training with Masked AutoEncoder (MAE) for LaneSegNet.
Uses unlabeled aerial imagery to learn rich visual representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""
    
    def __init__(self, img_size=1280, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone.
    Based on MAE paper: "Masked Autoencoders Are Scalable Vision Learners"
    """
    
    def __init__(self, img_size=1280, patch_size=16, in_channels=3,
                 embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder blocks
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=encoder_num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=encoder_depth
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=decoder_depth
        )
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights following MAE paper."""
        # Initialize positional embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def random_masking(self, x):
        """
        Random masking strategy from MAE paper.
        Args:
            x: [B, N, D] where N is number of patches
        Returns:
            x_masked: visible patches only
            mask: binary mask (0 is keep, 1 is remove)
            ids_restore: indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Generate random noise and sort
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """Forward pass through encoder."""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x)
        
        # Encoder
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder."""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # No class token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Add positional embedding
        x = x_ + self.decoder_pos_embed
        
        # Decoder
        x = self.decoder(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss on masked patches only."""
        # Patchify target
        target = self.patchify(imgs)
        
        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, imgs):
        """Full forward pass."""
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MAETrainer:
    """Trainer for Masked Autoencoder pre-training."""
    
    def __init__(self, model, device, save_dir="work_dirs/ssl_pretraining"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer (following MAE paper)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1.5e-4,  # Base learning rate
            betas=(0.9, 0.95),
            weight_decay=0.05
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=0
        )
        
        self.training_log = []
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, images in enumerate(progress):
            images = images.to(self.device)
            
            # Forward pass
            loss, pred, mask = self.model(images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'training_log': self.training_log
        }
        
        checkpoint_path = self.save_dir / f"mae_checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "mae_best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path} (Loss: {loss:.4f})")
    
    def train(self, dataloader, epochs=100):
        """Full training loop."""
        logger.info(f"Starting MAE pre-training for {epochs} epochs")
        logger.info(f"Dataset size: {len(dataloader.dataset)}")
        logger.info(f"Batch size: {dataloader.batch_size}")
        
        self.best_loss = float('inf')
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, epoch + 1)
            
            self.training_log.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Check if this is the best model so far
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save a checkpoint periodically and always save the best model
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, avg_loss, is_best=is_best)
        
        # Save final model
        self.save_checkpoint(epochs, avg_loss, is_best=False)
        
        # Save training log
        log_path = self.save_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        logger.info(f"MAE pre-training completed! Models saved to {self.save_dir}")

# This file contains the core MAE model and trainer classes.
# For execution and orchestration, use scripts/run_ssl_pretraining.py
