#!/usr/bin/env python3
"""
Phase 4: Loss Function Behavior Analysis
Analyzes OHEM DiceFocal loss function behavior and training dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES

def create_synthetic_lane_data(batch_size=4, img_size=512):
    """Create synthetic lane data with known patterns."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    targets = torch.zeros(batch_size, img_size, img_size, dtype=torch.long).to(device)
    
    # Create lane patterns with different class distributions
    for i in range(batch_size):
        if i == 0:
            # Horizontal lanes
            targets[i, 200:220, :] = 1  # Class 1 lane
            targets[i, 300:320, :] = 2  # Class 2 lane
        elif i == 1:
            # Vertical lanes
            targets[i, :, 200:220] = 1
            targets[i, :, 300:320] = 2
        elif i == 2:
            # Diagonal lanes
            for j in range(img_size):
                if 200 <= j <= 220:
                    targets[i, j, j] = 1
                if 300 <= j <= 320:
                    targets[i, j, img_size-1-j] = 2
        else:
            # Sparse lanes (class imbalance test)
            targets[i, 250:260, 100:400] = 1  # Small lane
            targets[i, 260:265, 150:350] = 2  # Very small lane
    
    return images, targets

def test_loss_function_behavior():
    """Test loss function behavior with synthetic data."""
    logger.info("🔍 Testing loss function behavior...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Import loss functions
        from scripts.ohem_loss import OHEMDiceFocalLoss
        
        # Test different loss configurations
        loss_configs = [
            {
                'name': 'Standard OHEM DiceFocal',
                'loss': OHEMDiceFocalLoss(
                    alpha=0.25, gamma=2.0, smooth=1e-6,
                    ohem_thresh=0.7, ohem_min_kept=512,
                    dice_weight=0.6, focal_weight=0.4
                )
            },
            {
                'name': 'High Gamma DiceFocal', 
                'loss': OHEMDiceFocalLoss(
                    alpha=0.25, gamma=4.0, smooth=1e-6,
                    ohem_thresh=0.7, ohem_min_kept=512,
                    dice_weight=0.6, focal_weight=0.4
                )
            },
            {
                'name': 'Dice-Heavy Loss',
                'loss': OHEMDiceFocalLoss(
                    alpha=0.25, gamma=2.0, smooth=1e-6,
                    ohem_thresh=0.7, ohem_min_kept=512,
                    dice_weight=0.8, focal_weight=0.2
                )
            },
            {
                'name': 'Standard CrossEntropy',
                'loss': nn.CrossEntropyLoss()
            }
        ]
        
        # Create synthetic data
        images, targets = create_synthetic_lane_data(batch_size=4)
        
        # Create model for prediction
        from scripts.run_finetuning import PretrainedLaneNet
        model = PretrainedLaneNet(num_classes=NUM_CLASSES, img_size=512).to(device)
        
        # Test each loss function
        loss_results = {}
        
        for config in loss_configs:
            logger.info(f"Testing: {config['name']}")
            
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                
                # Calculate loss
                if isinstance(config['loss'], OHEMDiceFocalLoss):
                    loss_value = config['loss'](outputs, targets)
                else:
                    loss_value = config['loss'](outputs, targets)
                
                logger.info(f"  Loss value: {loss_value.item():.4f}")
                
                # Analyze predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Calculate class distribution in predictions
                pred_classes = torch.unique(predictions, return_counts=True)
                logger.info(f"  Predicted classes: {pred_classes[0].tolist()}")
                logger.info(f"  Prediction counts: {pred_classes[1].tolist()}")
                
                # Calculate target class distribution
                target_classes = torch.unique(targets, return_counts=True)
                logger.info(f"  Target classes: {target_classes[0].tolist()}")
                logger.info(f"  Target counts: {target_classes[1].tolist()}")
                
                # Calculate simple IoU
                ious = []
                for class_id in range(1, NUM_CLASSES):
                    pred_mask = (predictions == class_id)
                    target_mask = (targets == class_id)
                    
                    intersection = (pred_mask & target_mask).sum().item()
                    union = (pred_mask | target_mask).sum().item()
                    
                    if union > 0:
                        iou = intersection / union
                        ious.append(iou)
                        logger.info(f"  Class {class_id} IoU: {iou:.3f}")
                
                mean_iou = np.mean(ious) if ious else 0.0
                logger.info(f"  Mean IoU: {mean_iou:.3f}")
                
                loss_results[config['name']] = {
                    'loss_value': loss_value.item(),
                    'mean_iou': mean_iou,
                    'predicted_classes': pred_classes[0].tolist(),
                    'prediction_counts': pred_classes[1].tolist()
                }
                
            except Exception as e:
                logger.error(f"  Error with {config['name']}: {e}")
                loss_results[config['name']] = {'error': str(e)}
        
        return loss_results
        
    except Exception as e:
        logger.error(f"Loss function testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_gradient_behavior():
    """Test gradient computation and magnitudes."""
    logger.info("📊 Testing gradient behavior...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model and data
        from scripts.run_finetuning import PretrainedLaneNet
        model = PretrainedLaneNet(num_classes=NUM_CLASSES, img_size=512).to(device)
        model.train()
        
        images, targets = create_synthetic_lane_data(batch_size=2)  # Smaller batch for gradient analysis
        
        # Test with different loss functions
        from scripts.ohem_loss import OHEMDiceFocalLoss
        
        losses_to_test = [
            ('CrossEntropy', nn.CrossEntropyLoss()),
            ('OHEM DiceFocal', OHEMDiceFocalLoss(
                alpha=0.25, gamma=2.0, smooth=1e-6,
                ohem_thresh=0.7, ohem_min_kept=256,
                dice_weight=0.6, focal_weight=0.4
            ))
        ]
        
        gradient_results = {}
        
        for loss_name, criterion in losses_to_test:
            logger.info(f"Testing gradients with: {loss_name}")
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            logger.info(f"  Loss: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Analyze gradients
            gradient_stats = {}
            total_params = 0
            zero_grad_params = 0
            gradient_norms = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_max = param.grad.abs().max().item()
                    grad_mean = param.grad.abs().mean().item()
                    
                    gradient_norms.append(grad_norm)
                    total_params += 1
                    
                    if grad_norm < 1e-10:
                        zero_grad_params += 1
                    
                    # Log significant gradients
                    if grad_norm > 1e-6:
                        gradient_stats[name] = {
                            'norm': grad_norm,
                            'max': grad_max,
                            'mean': grad_mean
                        }
            
            # Overall gradient statistics
            if gradient_norms:
                avg_grad_norm = np.mean(gradient_norms)
                max_grad_norm = np.max(gradient_norms)
                min_grad_norm = np.min(gradient_norms)
                
                logger.info(f"  Gradient norms - Avg: {avg_grad_norm:.6f}, Max: {max_grad_norm:.6f}, Min: {min_grad_norm:.6f}")
                logger.info(f"  Parameters with gradients: {total_params - zero_grad_params}/{total_params}")
                
                # Check for gradient issues
                if max_grad_norm > 100:
                    logger.warning(f"  ⚠️ Large gradients detected (max: {max_grad_norm:.2f})")
                if avg_grad_norm < 1e-8:
                    logger.warning(f"  ⚠️ Very small gradients (avg: {avg_grad_norm:.2e})")
                if zero_grad_params > total_params * 0.5:
                    logger.warning(f"  ⚠️ Too many zero gradients ({zero_grad_params}/{total_params})")
            
            gradient_results[loss_name] = {
                'loss_value': loss.item(),
                'avg_grad_norm': avg_grad_norm if gradient_norms else 0,
                'max_grad_norm': max_grad_norm if gradient_norms else 0,
                'zero_grad_ratio': zero_grad_params / total_params if total_params > 0 else 1,
                'top_gradients': dict(sorted(gradient_stats.items(), 
                                           key=lambda x: x[1]['norm'], reverse=True)[:5])
            }
        
        return gradient_results
        
    except Exception as e:
        logger.error(f"Gradient behavior testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_class_imbalance_handling():
    """Test how loss functions handle class imbalance."""
    logger.info("⚖️ Testing class imbalance handling...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create severely imbalanced data
        images = torch.randn(2, 3, 256, 256).to(device)  # Smaller for faster testing
        targets = torch.zeros(2, 256, 256, dtype=torch.long).to(device)
        
        # Create extreme class imbalance: 99% background, <1% lanes
        targets[0, 100:110, 100:150] = 1  # Small lane segment
        targets[1, 120:125, 120:140] = 2  # Very small lane segment
        
        # Calculate class distribution
        unique, counts = torch.unique(targets, return_counts=True)
        total_pixels = targets.numel()
        
        logger.info(f"Class distribution:")
        for cls, count in zip(unique.tolist(), counts.tolist()):
            percentage = count / total_pixels * 100
            logger.info(f"  Class {cls}: {count} pixels ({percentage:.2f}%)")
        
        # Test different loss functions on imbalanced data
        from scripts.ohem_loss import OHEMDiceFocalLoss
        
        loss_tests = [
            ('Standard CrossEntropy', nn.CrossEntropyLoss()),
            ('Weighted CrossEntropy', nn.CrossEntropyLoss(weight=torch.tensor([0.1, 5.0, 5.0]).to(device))),
            ('OHEM DiceFocal', OHEMDiceFocalLoss(
                alpha=0.25, gamma=2.0, smooth=1e-6,
                ohem_thresh=0.7, ohem_min_kept=128,
                dice_weight=0.6, focal_weight=0.4
            ))
        ]
        
        # Create simple model for testing
        simple_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, NUM_CLASSES, 1)
        ).to(device)
        
        imbalance_results = {}
        
        for loss_name, criterion in loss_tests:
            logger.info(f"Testing: {loss_name}")
            
            simple_model.eval()
            with torch.no_grad():
                outputs = simple_model(images)
            
            # Calculate loss
            loss_value = criterion(outputs, targets)
            logger.info(f"  Loss: {loss_value.item():.4f}")
            
            # Check predictions
            predictions = torch.argmax(outputs, dim=1)
            pred_unique, pred_counts = torch.unique(predictions, return_counts=True)
            
            logger.info(f"  Predicted classes: {pred_unique.tolist()}")
            
            # Calculate per-class IoU
            class_ious = {}
            for class_id in range(1, NUM_CLASSES):
                pred_mask = (predictions == class_id)
                target_mask = (targets == class_id)
                
                intersection = (pred_mask & target_mask).sum().item()
                union = (pred_mask | target_mask).sum().item()
                
                if union > 0:
                    iou = intersection / union
                    class_ious[class_id] = iou
                    logger.info(f"  Class {class_id} IoU: {iou:.3f}")
                else:
                    class_ious[class_id] = 0.0
                    logger.info(f"  Class {class_id}: No ground truth")
            
            imbalance_results[loss_name] = {
                'loss_value': loss_value.item(),
                'class_ious': class_ious,
                'predicted_classes': pred_unique.tolist()
            }
        
        return imbalance_results
        
    except Exception as e:
        logger.error(f"Class imbalance testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_ohem_parameters():
    """Analyze OHEM parameters and their effects."""
    logger.info("🔧 Analyzing OHEM parameters...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from scripts.ohem_loss import OHEMDiceFocalLoss
        
        # Create test data
        images, targets = create_synthetic_lane_data(batch_size=2, img_size=256)
        
        # Create model
        from scripts.run_finetuning import PretrainedLaneNet
        model = PretrainedLaneNet(num_classes=NUM_CLASSES, img_size=256).to(device)
        
        # Test different OHEM configurations
        ohem_configs = [
            {
                'name': 'Conservative OHEM',
                'ohem_thresh': 0.9,
                'ohem_min_kept': 1024,
                'dice_weight': 0.5,
                'focal_weight': 0.5
            },
            {
                'name': 'Aggressive OHEM', 
                'ohem_thresh': 0.5,
                'ohem_min_kept': 256,
                'dice_weight': 0.7,
                'focal_weight': 0.3
            },
            {
                'name': 'Current Settings',
                'ohem_thresh': 0.7,
                'ohem_min_kept': 512,
                'dice_weight': 0.6,
                'focal_weight': 0.4
            }
        ]
        
        ohem_results = {}
        
        for config in ohem_configs:
            logger.info(f"Testing: {config['name']}")
            
            # Create loss function
            loss_fn = OHEMDiceFocalLoss(
                alpha=0.25,
                gamma=2.0,
                smooth=1e-6,
                ohem_thresh=config['ohem_thresh'],
                ohem_min_kept=config['ohem_min_kept'],
                dice_weight=config['dice_weight'],
                focal_weight=config['focal_weight']
            )
            
            model.eval()
            with torch.no_grad():
                outputs = model(images)
            
            # Calculate loss
            loss_value = loss_fn(outputs, targets)
            logger.info(f"  Loss: {loss_value.item():.4f}")
            logger.info(f"  OHEM thresh: {config['ohem_thresh']}")
            logger.info(f"  Min kept: {config['ohem_min_kept']}")
            logger.info(f"  Dice/Focal weights: {config['dice_weight']:.1f}/{config['focal_weight']:.1f}")
            
            ohem_results[config['name']] = {
                'loss_value': loss_value.item(),
                'config': config
            }
        
        return ohem_results
        
    except Exception as e:
        logger.error(f"OHEM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def generate_loss_analysis_report(loss_results, gradient_results, imbalance_results, ohem_results):
    """Generate comprehensive loss analysis report."""
    report = {
        "test_type": "loss_function_behavior_analysis",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "loss_function_tests": loss_results,
        "gradient_analysis": gradient_results,
        "class_imbalance_tests": imbalance_results,
        "ohem_parameter_analysis": ohem_results,
        "critical_issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Analyze for critical issues
    if gradient_results:
        for loss_name, grad_result in gradient_results.items():
            if grad_result.get('avg_grad_norm', 0) < 1e-8:
                report["critical_issues"].append(f"Very small gradients with {loss_name}")
            if grad_result.get('zero_grad_ratio', 1) > 0.8:
                report["critical_issues"].append(f"Too many zero gradients with {loss_name}")
    
    # Check for class imbalance issues
    if imbalance_results:
        for loss_name, imb_result in imbalance_results.items():
            class_ious = imb_result.get('class_ious', {})
            if all(iou < 0.01 for iou in class_ious.values()):
                report["critical_issues"].append(f"Cannot handle class imbalance with {loss_name}")
    
    # Generate recommendations
    if not report["critical_issues"]:
        report["recommendations"].append("Loss function analysis shows no critical issues")
        report["recommendations"].append("Consider fine-tuning OHEM parameters or learning rates")
    else:
        report["recommendations"].append("Switch to different loss function or adjust parameters")
    
    # Save report
    report_path = Path("work_dirs/loss_analysis_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Loss analysis report saved: {report_path}")
    return report

def main():
    """Run Phase 4: Loss function behavior analysis."""
    logger.info("🚀 PHASE 4: LOSS FUNCTION BEHAVIOR ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Test 1: Loss Function Behavior
        logger.info("\n🔍 TESTING LOSS FUNCTION BEHAVIOR")
        logger.info("-" * 40)
        loss_results = test_loss_function_behavior()
        
        # Test 2: Gradient Behavior
        logger.info("\n📊 TESTING GRADIENT BEHAVIOR")
        logger.info("-" * 40)
        gradient_results = test_gradient_behavior()
        
        # Test 3: Class Imbalance Handling
        logger.info("\n⚖️ TESTING CLASS IMBALANCE HANDLING")
        logger.info("-" * 40)
        imbalance_results = test_class_imbalance_handling()
        
        # Test 4: OHEM Parameter Analysis
        logger.info("\n🔧 ANALYZING OHEM PARAMETERS")
        logger.info("-" * 40)
        ohem_results = analyze_ohem_parameters()
        
        # Generate comprehensive report
        report = generate_loss_analysis_report(
            loss_results, gradient_results, imbalance_results, ohem_results
        )
        
        # Final Assessment
        logger.info("\n" + "=" * 60)
        logger.info("📊 LOSS FUNCTION ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        if report["critical_issues"]:
            logger.error("❌ CRITICAL LOSS FUNCTION ISSUES FOUND:")
            for issue in report["critical_issues"]:
                logger.error(f"  • {issue}")
            logger.error("🚫 LOSS FUNCTION HAS PROBLEMS")
        else:
            logger.info("✅ NO CRITICAL LOSS FUNCTION ISSUES FOUND")
            logger.info("🎯 LOSS FUNCTION APPEARS FUNCTIONAL")
            
        logger.info(f"\n📋 Recommendations:")
        for rec in report.get("recommendations", []):
            logger.info(f"  • {rec}")
        
        return len(report["critical_issues"]) == 0
        
    except Exception as e:
        logger.error(f"💥 Loss function analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)