#!/usr/bin/env python3
"""
Quick diagnostic script to identify critical training infrastructure issues.
Based on Gemini's analysis, this will test the core problems:
1. Training completion issues
2. Loss function failures
3. Class collapse problems
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.configurable_finetuning import PretrainedLaneNet, LaneSegDataset
from scripts.ohem_loss import OHEMDiceFocalLoss
from configs.global_config import NUM_CLASSES

class TrainingDiagnostics:
    """Quick diagnostics for training infrastructure issues"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def test_data_loading(self):
        """Test data loading and class distribution"""
        print("\n=== DATA LOADING TEST ===")
        
        # Test dataset loading
        try:
            train_dataset = LaneSegDataset(
                'data/ael_mmseg/img_dir/train',
                'data/ael_mmseg/ann_dir/train',
                mode='train'
            )
            
            val_dataset = LaneSegDataset(
                'data/ael_mmseg/img_dir/val',
                'data/ael_mmseg/ann_dir/val',
                mode='val'
            )
            
            print(f"✓ Train dataset: {len(train_dataset)} samples")
            print(f"✓ Val dataset: {len(val_dataset)} samples")
            
            # Test a few samples and check class distribution
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
            
            class_counts = np.zeros(NUM_CLASSES)
            total_pixels = 0
            samples_checked = 0
            
            for images, masks in train_loader:
                if samples_checked >= 10:  # Check first 10 batches
                    break
                    
                # Count class distribution
                for mask in masks:
                    mask_np = mask.numpy()
                    for class_id in range(NUM_CLASSES):
                        class_counts[class_id] += (mask_np == class_id).sum()
                    total_pixels += mask_np.size
                    
                samples_checked += 1
            
            # Print class distribution
            print(f"\nClass distribution (first {samples_checked * 4} samples):")
            for class_id in range(NUM_CLASSES):
                percentage = (class_counts[class_id] / total_pixels) * 100
                print(f"  Class {class_id}: {percentage:.2f}% ({int(class_counts[class_id])} pixels)")
            
            # Check for class imbalance severity
            max_class_ratio = max(class_counts) / min(class_counts[class_counts > 0])
            print(f"  Max class imbalance ratio: {max_class_ratio:.1f}:1")
            
            if max_class_ratio > 100:
                print("  ⚠️  SEVERE CLASS IMBALANCE DETECTED")
            
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def test_model_creation(self):
        """Test model creation and basic forward pass"""
        print("\n=== MODEL CREATION TEST ===")
        
        try:
            # Test model creation
            model = PretrainedLaneNet(
                num_classes=NUM_CLASSES,
                img_size=512,
                encoder_weights_path="work_dirs/mae_pretraining/mae_best_model.pth"
            ).to(self.device)
            
            print("✓ Model created successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ Total parameters: {total_params:,}")
            print(f"✓ Trainable parameters: {trainable_params:,}")
            
            # Test forward pass
            test_input = torch.randn(2, 3, 512, 512).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            forward_time = time.time() - start_time
            
            print(f"✓ Forward pass successful: {test_input.shape} → {output.shape}")
            print(f"✓ Forward pass time: {forward_time:.3f}s for batch of 2")
            
            # Check output values
            output_min, output_max = output.min().item(), output.max().item()
            print(f"✓ Output range: [{output_min:.3f}, {output_max:.3f}]")
            
            # Test prediction
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Check if model predicts multiple classes
            unique_preds = torch.unique(preds)
            print(f"✓ Model predicts {len(unique_preds)} unique classes: {unique_preds.cpu().numpy()}")
            
            if len(unique_preds) == 1:
                print("  ⚠️  MODEL ONLY PREDICTS ONE CLASS - POTENTIAL ISSUE")
            
            return model
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return None
    
    def test_loss_function(self, model):
        """Test loss function behavior"""
        print("\n=== LOSS FUNCTION TEST ===")
        
        if model is None:
            print("❌ Skipping loss test - no model available")
            return
        
        try:
            # Create test data
            batch_size = 2
            test_input = torch.randn(batch_size, 3, 512, 512).to(self.device)
            test_target = torch.randint(0, NUM_CLASSES, (batch_size, 512, 512)).to(self.device)
            
            # Test different loss configurations
            loss_configs = [
                {"use_ohem": False, "description": "Standard CrossEntropy"},
                {"use_ohem": True, "ohem_alpha": 0.25, "ohem_gamma": 2.0, "description": "OHEM DiceFocal"},
                {"use_ohem": True, "ohem_alpha": 0.1, "ohem_gamma": 5.0, "description": "Aggressive OHEM"},
            ]
            
            for config in loss_configs:
                print(f"\nTesting: {config['description']}")
                
                if config["use_ohem"]:
                    loss_fn = OHEMDiceFocalLoss(
                        ohem_alpha=config["ohem_alpha"],
                        ohem_gamma=config["ohem_gamma"],
                        dice_weight=0.6,
                        focal_weight=0.4
                    ).to(self.device)
                else:
                    loss_fn = torch.nn.CrossEntropyLoss()
                
                # Forward pass
                output = model(test_input)
                loss = loss_fn(output, test_target)
                
                print(f"  Loss value: {loss.item():.4f}")
                
                # Check gradients
                loss.backward()
                
                # Check if gradients exist and are reasonable
                total_grad_norm = 0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    avg_grad_norm = (total_grad_norm / param_count) ** 0.5
                    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
                    
                    if avg_grad_norm < 1e-8:
                        print("  ⚠️  VERY SMALL GRADIENTS - POTENTIAL VANISHING GRADIENT")
                    elif avg_grad_norm > 10:
                        print("  ⚠️  LARGE GRADIENTS - POTENTIAL EXPLODING GRADIENT")
                else:
                    print("  ❌ NO GRADIENTS COMPUTED")
                
                # Clear gradients for next test
                model.zero_grad()
            
            return True
            
        except Exception as e:
            print(f"❌ Loss function test failed: {e}")
            return False
    
    def test_training_step(self, model):
        """Test a single training step for timing and memory issues"""
        print("\n=== TRAINING STEP TEST ===")
        
        if model is None:
            print("❌ Skipping training step test - no model available")
            return
        
        try:
            # Create mini dataset
            train_dataset = LaneSegDataset(
                'data/ael_mmseg/img_dir/train',
                'data/ael_mmseg/ann_dir/train',
                mode='train'
            )
            
            # Small batch for testing
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
            
            # Setup optimizer and loss
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss_fn = OHEMDiceFocalLoss().to(self.device)
            
            model.train()
            
            print("Testing 5 training steps...")
            step_times = []
            
            for i, (images, masks) in enumerate(train_loader):
                if i >= 5:  # Test only 5 steps
                    break
                
                start_time = time.time()
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step_time = time.time() - start_time
                step_times.append(step_time)
                
                print(f"  Step {i+1}: {step_time:.2f}s, Loss: {loss.item():.4f}")
                
                # Check memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"           GPU Memory: {memory_mb:.1f}MB")
            
            avg_step_time = np.mean(step_times)
            print(f"\n✓ Average step time: {avg_step_time:.2f}s")
            
            # Estimate time for full epoch
            steps_per_epoch = len(train_loader)
            estimated_epoch_time = avg_step_time * steps_per_epoch / 60  # minutes
            print(f"✓ Estimated epoch time: {estimated_epoch_time:.1f} minutes")
            
            if estimated_epoch_time > 60:
                print("  ⚠️  VERY SLOW TRAINING - EACH EPOCH WOULD TAKE >1 HOUR")
            
            return True
            
        except Exception as e:
            print(f"❌ Training step test failed: {e}")
            return False
    
    def run_full_diagnostics(self):
        """Run all diagnostic tests"""
        print("=" * 60)
        print("TRAINING INFRASTRUCTURE DIAGNOSTICS")
        print("=" * 60)
        
        # Test 1: Data loading
        data_ok = self.test_data_loading()
        
        # Test 2: Model creation
        model = self.test_model_creation()
        
        # Test 3: Loss function
        loss_ok = self.test_loss_function(model)
        
        # Test 4: Training step timing
        training_ok = self.test_training_step(model)
        
        # Summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Data Loading: {'✓ PASS' if data_ok else '❌ FAIL'}")
        print(f"Model Creation: {'✓ PASS' if model is not None else '❌ FAIL'}")
        print(f"Loss Function: {'✓ PASS' if loss_ok else '❌ FAIL'}")
        print(f"Training Steps: {'✓ PASS' if training_ok else '❌ FAIL'}")
        
        if all([data_ok, model is not None, loss_ok, training_ok]):
            print("\n🎉 ALL TESTS PASSED - Infrastructure appears stable")
        else:
            print("\n⚠️  ISSUES DETECTED - Infrastructure needs fixes")

if __name__ == "__main__":
    diagnostics = TrainingDiagnostics()
    diagnostics.run_full_diagnostics()