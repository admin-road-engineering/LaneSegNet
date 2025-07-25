#!/usr/bin/env python3
"""
Phase 1: Overfitting Test on Tiny Dataset
CRITICAL DECISION POINT: Test if model can achieve >90% IoU on 10-20 samples.

SUCCESS → Problem is in full dataset/validation (go to Phase 2)
FAILURE → Problem is in core training loop (go to Phase 3)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TinyDatasetOverfitter:
    """Specialized trainer for tiny dataset overfitting test."""
    
    def __init__(self, model, device, tiny_dataset, max_epochs=50):
        self.model = model.to(device)
        self.device = device
        self.tiny_dataset = tiny_dataset
        self.max_epochs = max_epochs
        
        # Aggressive optimizer for fast overfitting
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Simple CrossEntropy loss for overfitting test
        self.criterion = nn.CrossEntropyLoss()
        
        # Track metrics
        self.training_history = []
        
    def calculate_iou(self, pred_tensor, target_tensor):
        """Calculate IoU for the tiny dataset."""
        pred_np = pred_tensor.cpu().numpy()
        target_np = target_tensor.cpu().numpy()
        
        # Use model's actual number of classes
        num_classes = self.model.final_conv.out_channels
        
        ious = []
        for class_id in range(1, num_classes):  # Skip background (0)
            pred_mask = (pred_np == class_id)
            target_mask = (target_np == class_id)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union == 0:
                # No ground truth for this class in tiny dataset - skip
                continue
            else:
                iou = intersection / union
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def train_single_epoch(self, epoch):
        """Train one epoch on tiny dataset."""
        self.model.train()
        
        # Create dataloader with tiny dataset (batch_size=1 for maximum overfitting)
        dataloader = DataLoader(self.tiny_dataset, batch_size=1, shuffle=True)
        
        epoch_loss = 0
        epoch_iou = 0
        num_batches = 0
        
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate IoU
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                iou = self.calculate_iou(predictions, targets)
            
            epoch_loss += loss.item()
            epoch_iou += iou
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_iou = epoch_iou / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_iou
    
    def run_overfitting_test(self):
        """Run the complete overfitting test."""
        logger.info(f"🎯 Starting overfitting test on {len(self.tiny_dataset)} samples")
        logger.info(f"Target: >90% IoU within {self.max_epochs} epochs")
        
        best_iou = 0
        early_success = False
        
        for epoch in range(self.max_epochs):
            loss, iou = self.train_single_epoch(epoch)
            
            # Track progress
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': loss,
                'iou': iou
            })
            
            # Update best IoU
            if iou > best_iou:
                best_iou = iou
            
            # Log progress
            logger.info(f"Epoch {epoch+1:2d}/{self.max_epochs}: Loss={loss:.4f}, IoU={iou:.1%} (best: {best_iou:.1%})")
            
            # Check for early success
            if iou >= 0.90:  # 90% IoU threshold
                logger.info(f"🎉 SUCCESS! Achieved {iou:.1%} IoU at epoch {epoch+1}")
                early_success = True
                break
            
            # Check for reasonable progress
            if epoch >= 10 and best_iou < 0.10:  # Less than 10% after 10 epochs
                logger.warning(f"⚠️ Poor progress: Only {best_iou:.1%} IoU after {epoch+1} epochs")
        
        # Final assessment
        success = best_iou >= 0.90
        
        logger.info("=" * 60)
        logger.info("🏁 OVERFITTING TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best IoU achieved: {best_iou:.1%}")
        logger.info(f"Target IoU (90%): {'✅ ACHIEVED' if success else '❌ FAILED'}")
        logger.info(f"Early success: {'Yes' if early_success else 'No'}")
        
        return success, best_iou, self.training_history

def create_tiny_dataset(base_dataset, num_samples=15):
    """Create a tiny subset of the dataset for overfitting test."""
    logger.info(f"🔍 Creating tiny dataset with {num_samples} samples...")
    
    # Select samples with good class distribution
    indices_with_lanes = []
    indices_checked = 0
    max_check = min(1000, len(base_dataset))  # Don't check entire dataset
    
    for idx in range(max_check):
        try:
            _, mask = base_dataset[idx]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            
            # Check if this sample has lane pixels (non-zero, non-background)
            unique_classes = np.unique(mask_np)
            has_lanes = any(cls > 0 for cls in unique_classes)
            
            if has_lanes:
                indices_with_lanes.append(idx)
                
            indices_checked += 1
            
            # Stop when we have enough good samples
            if len(indices_with_lanes) >= num_samples:
                break
                
        except Exception as e:
            logger.warning(f"Error checking sample {idx}: {e}")
            continue
    
    if len(indices_with_lanes) < num_samples:
        logger.warning(f"Only found {len(indices_with_lanes)} samples with lanes (wanted {num_samples})")
        # Use what we found
        selected_indices = indices_with_lanes
    else:
        # Select the first num_samples good samples
        selected_indices = indices_with_lanes[:num_samples]
    
    logger.info(f"✅ Selected {len(selected_indices)} samples with lane markings")
    
    # Create subset
    tiny_dataset = Subset(base_dataset, selected_indices)
    
    # Log sample info
    logger.info("📊 Tiny dataset sample analysis:")
    for i, global_idx in enumerate(selected_indices[:5]):  # Show first 5
        try:
            _, mask = base_dataset[global_idx]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            unique_classes = np.unique(mask_np)
            logger.info(f"  Sample {i}: Classes {unique_classes.tolist()}")
        except Exception as e:
            logger.warning(f"  Sample {i}: Error analyzing - {e}")
    
    return tiny_dataset

def analyze_test_results(success, best_iou, history):
    """Analyze test results and provide next steps."""
    logger.info("🔍 ANALYZING TEST RESULTS...")
    
    result = {
        "test_type": "tiny_dataset_overfitting",
        "success": success,
        "best_iou": best_iou,
        "target_iou": 0.90,
        "epochs_run": len(history),
        "final_loss": history[-1]["loss"] if history else None,
        "final_iou": history[-1]["iou"] if history else None,
        "training_history": history
    }
    
    if success:
        logger.info("✅ PHASE 1 SUCCESS: Model CAN overfit tiny dataset")
        logger.info("🎯 CONCLUSION: Problem is in FULL DATASET or VALIDATION")
        logger.info("📋 NEXT STEPS:")
        logger.info("  → Proceed to Phase 2: Data Integrity Deep Dive")
        logger.info("  → Focus on: Dataset quality, class distribution, data loading")
        logger.info("  → Time estimate: 1-2 days")
        
        result["conclusion"] = "data_validation_issue"
        result["next_phase"] = "phase_2_data_integrity"
        result["recommended_actions"] = [
            "Validate ground truth quality",
            "Analyze class distribution across full dataset", 
            "Audit data loading pipeline",
            "Check for label corruption or mismatch"
        ]
        
    else:
        logger.error("❌ PHASE 1 FAILURE: Model CANNOT overfit tiny dataset")
        logger.error("🎯 CONCLUSION: Problem is in CORE TRAINING LOOP")
        logger.error("📋 NEXT STEPS:")
        logger.error("  → Proceed to Phase 3: Model Architecture Validation")
        logger.error("  → Focus on: Model architecture, forward pass, loss function")
        logger.error("  → Time estimate: 1-2 days")
        
        result["conclusion"] = "core_training_issue"
        result["next_phase"] = "phase_3_architecture_validation"
        result["recommended_actions"] = [
            "Validate model forward pass",
            "Check weight loading and initialization",
            "Analyze loss function behavior",
            "Verify gradient flow through model"
        ]
        
        # Additional diagnostics for failure case
        if best_iou < 0.05:
            logger.error("🚨 SEVERE: IoU < 5% suggests fundamental architecture issue")
            result["severity"] = "critical_architecture_issue"
        elif best_iou < 0.30:
            logger.error("⚠️ MODERATE: IoU < 30% suggests loss function or optimization issue")
            result["severity"] = "optimization_issue"
        else:
            logger.warning("💡 MILD: IoU < 90% may be insufficient training or learning rate")
            result["severity"] = "hyperparameter_issue"
    
    # Save detailed results
    results_path = Path("work_dirs/overfitting_test_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"📄 Detailed results saved: {results_path}")
    
    return result

def main():
    """Run Phase 1: Overfitting test on tiny dataset."""
    logger.info("🚀 PHASE 1: OVERFITTING TEST (CRITICAL DECISION POINT)")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Import and create model
        logger.info("🏗️ Creating model...")
        from scripts.run_finetuning import PretrainedLaneNet
        
        model = PretrainedLaneNet(
            num_classes=NUM_CLASSES,
            img_size=512,
            encoder_weights_path=None,  # Start with random weights for fair test
            freeze_encoder=False
        )
        
        # Create base dataset
        logger.info("📁 Loading base dataset...")
        from data.labeled_dataset import LabeledDataset
        
        base_dataset = LabeledDataset(
            "data/ael_mmseg/img_dir/train",
            "data/ael_mmseg/ann_dir/train", 
            mode='train',
            img_size=(512, 512)
        )
        
        logger.info(f"Base dataset size: {len(base_dataset)} samples")
        
        # Create tiny dataset
        tiny_dataset = create_tiny_dataset(base_dataset, num_samples=15)
        
        # Create overfitter
        overfitter = TinyDatasetOverfitter(
            model=model,
            device=device,
            tiny_dataset=tiny_dataset,
            max_epochs=50
        )
        
        # Run the test
        logger.info("🎯 Starting overfitting test...")
        start_time = time.time()
        
        success, best_iou, history = overfitter.run_overfitting_test()
        
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        logger.info(f"⏱️ Test completed in {duration_minutes:.1f} minutes")
        
        # Analyze results and determine next steps
        result = analyze_test_results(success, best_iou, history)
        
        logger.info("=" * 70)
        logger.info("🎯 PHASE 1 COMPLETE - DECISION MADE")
        logger.info("=" * 70)
        
        return success
        
    except Exception as e:
        logger.error(f"💥 Overfitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)