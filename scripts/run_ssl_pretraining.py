#!/usr/bin/env python3
"""
Self-Supervised Pre-training with Masked Autoencoder (MAE)
===========================================================

This script runs the pre-training phase on the collected unlabeled aerial imagery.
It uses the Masked Autoencoder (MAE) architecture to learn robust visual 
representations.

The resulting pre-trained encoder weights can then be used as a superior
starting point for the downstream lane detection task, which should significantly
improve generalization and performance on the test set.

Usage:
    python scripts/run_ssl_pretraining.py --epochs 100 --batch-size 32 --lr 1.5e-4
"""

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our implementations
from scripts.ssl_pretraining import MaskedAutoencoderViT, MAETrainer
from data.unlabeled_dataset import ConsolidatedUnlabeledDataset

def check_data_availability(data_dir):
    """Check if unlabeled data is available."""
    print("=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)
    
    # Check different data sources
    sources = {
        'consolidated': os.path.join(data_dir, 'consolidated'),
        'osm_1000': os.path.join(data_dir, 'osm_1000'),
        'osm_test': os.path.join(data_dir, 'osm_test'),
        'skyscapes': os.path.join(data_dir, 'skyscapes', 'processed'),
        'carla': os.path.join(data_dir, 'carla_synthetic', 'processed'),
        'cityscapes': os.path.join(data_dir, 'cityscapes_aerial', 'processed')
    }
    
    total_images = 0
    available_sources = []
    
    for source_name, source_path in sources.items():
        if os.path.exists(source_path):
            # Count images
            image_count = 0
            for root, dirs, files in os.walk(source_path):
                image_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if image_count > 0:
                print(f"✓ {source_name:<12}: {image_count:>4} images")
                total_images += image_count
                available_sources.append(source_name)
            else:
                print(f"✗ {source_name:<12}: {image_count:>4} images (empty)")
        else:
            print(f"✗ {source_name:<12}:  N/A images (not found)")
    
    print("-" * 60)
    print(f"TOTAL AVAILABLE: {total_images} images from {len(available_sources)} sources")
    
    # SSL readiness assessment
    if total_images >= 1000:
        print("🟢 EXCELLENT: 1000+ images - Full SSL pre-training recommended")
        return True, 'excellent'
    elif total_images >= 500:
        print("🟡 GOOD: 500+ images - SSL pre-training will be effective")
        return True, 'good'
    elif total_images >= 100:
        print("🟠 LIMITED: 100+ images - Basic SSL demo possible")
        return True, 'limited'
    else:
        print("🔴 INSUFFICIENT: <100 images - Run data collection first")
        return False, 'insufficient'

def create_mae_model(args):
    """Create MAE model with specified parameters."""
    model = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MAE Model Created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def create_dataloader(args):
    """Create dataloader for unlabeled aerial imagery."""
    
    # Transforms for SSL pre-training
    # Keep minimal - MAE handles its own augmentation via masking
    transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Create dataset
    dataset = ConsolidatedUnlabeledDataset(
        base_dir=args.data_dir,
        transform=transform,
        prefer_consolidated=True
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.data_dir}. Run data collection first.")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"Dataloader Created:")
    print(f"  Dataset size: {len(dataset):,} images")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {len(dataloader):,}")
    print(f"  Workers: {args.num_workers}")
    
    return dataloader

def save_training_config(args, save_dir):
    """Save training configuration for reproducibility."""
    config = {
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'img_size': args.img_size,
        'num_workers': args.num_workers,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'model_config': {
            'img_size': args.img_size,
            'patch_size': 16,
            'embed_dim': 768,
            'encoder_depth': 12,
            'decoder_depth': 8,
            'mask_ratio': 0.75
        }
    }
    
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training config saved: {config_path}")

def main(args):
    """Main pre-training orchestration."""
    
    print("=" * 80)
    print("MASKED AUTOENCODER (MAE) SELF-SUPERVISED PRE-TRAINING")
    print("=" * 80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 80)
    
    # Check data availability
    data_available, data_status = check_data_availability(args.data_dir)
    if not data_available:
        print("\n❌ INSUFFICIENT DATA FOR PRE-TRAINING")
        print("Please run data collection first:")
        print("1. run_data_collection.bat")
        print("2. python scripts/consolidate_unlabeled_data.py")
        return False
    
    # Adjust parameters based on data availability
    if data_status == 'limited':
        print(f"\n⚠️  ADJUSTING PARAMETERS FOR LIMITED DATA")
        args.epochs = min(args.epochs, 50)  # Reduce epochs for small datasets
        args.batch_size = min(args.batch_size, 8)  # Smaller batches
        print(f"Adjusted epochs: {args.epochs}")
        print(f"Adjusted batch size: {args.batch_size}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_training_config(args, save_dir)
    
    print(f"\n📁 Results will be saved to: {save_dir}")
    
    try:
        # Create model
        print(f"\n⚙️  CREATING MAE MODEL")
        model = create_mae_model(args)
        
        # Create dataloader  
        print(f"\n📊 PREPARING DATA")
        dataloader = create_dataloader(args)
        
        # Create trainer
        print(f"\n🚀 INITIALIZING TRAINER")
        trainer = MAETrainer(model, device, save_dir)
        
        # Start pre-training
        print(f"\n🎯 STARTING PRE-TRAINING ({args.epochs} epochs)")
        print("=" * 80)
        
        start_time = time.time()
        trainer.train(dataloader, epochs=args.epochs)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        print("=" * 80)
        print("✅ PRE-TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Training time: {training_time/3600:.2f} hours")
        print(f"💾 Models saved to: {save_dir}")
        print(f"📊 Best model: {save_dir}/mae_best_model.pth")
        print(f"📈 Training log: {save_dir}/training_log.json")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Use pre-trained encoder for lane detection fine-tuning")
        print("2. Load encoder weights: model.encoder.load_state_dict(...)")
        print("3. Expected improvement: +5-15% mIoU on test set")
        print("4. Better generalization across different cities/conditions")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  PRE-TRAINING INTERRUPTED BY USER")
        print(f"Partial results saved to: {save_dir}")
        return False
        
    except Exception as e:
        print(f"\n❌ PRE-TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAE Self-Supervised Pre-training")
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/unlabeled_aerial', 
                       help='Directory with unlabeled images')
    parser.add_argument('--img-size', type=int, default=224, 
                       help='Image size for training (224 for efficiency, 1280 for full resolution)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of pre-training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for pre-training')
    parser.add_argument('--lr', type=float, default=1.5e-4, 
                       help='Learning rate for AdamW optimizer')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='work_dirs/mae_pretraining',
                       help='Directory to save checkpoints and logs')
    
    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=10, 
                       help='How many steps to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=10, 
                       help='How many epochs to wait before saving a checkpoint')
    
    args = parser.parse_args()
    
    success = main(args)
    sys.exit(0 if success else 1)