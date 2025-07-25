#!/usr/bin/env python3
"""
Phase 3.1: Model Forward Pass Validation
Validates model architecture and forward pass with synthetic data.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES

def test_model_creation():
    """Test basic model creation and parameter validation."""
    logger.info("🏗️ Testing model creation...")
    
    try:
        from scripts.run_finetuning import PretrainedLaneNet
        
        # Test with standard parameters
        model = PretrainedLaneNet(
            num_classes=NUM_CLASSES,
            img_size=512,
            encoder_weights_path=None,
            freeze_encoder=False
        )
        
        # Validate architecture
        tests_passed = []
        
        # Test 1: Final layer has correct output channels
        final_channels = model.final_conv.out_channels
        if final_channels == NUM_CLASSES:
            logger.info(f"✅ Final layer channels: {final_channels} (matches NUM_CLASSES)")
            tests_passed.append("final_layer_channels")
        else:
            logger.error(f"❌ Final layer channels: {final_channels} (expected {NUM_CLASSES})")
        
        # Test 2: Model parameters are trainable
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 0:
            logger.info(f"✅ Trainable parameters: {total_params:,}")
            tests_passed.append("trainable_parameters")
        else:
            logger.error("❌ No trainable parameters found")
        
        # Test 3: Model components exist
        required_components = ['patch_embed', 'pos_embed', 'encoder', 'decoder', 'upsample_layers', 'final_conv']
        for component in required_components:
            if hasattr(model, component):
                logger.info(f"✅ Component exists: {component}")
                tests_passed.append(f"component_{component}")
            else:
                logger.error(f"❌ Missing component: {component}")
        
        return model, tests_passed
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def test_forward_pass_shapes(model):
    """Test forward pass with various input shapes."""
    logger.info("🔍 Testing forward pass shapes...")
    
    model.eval()
    tests_passed = []
    
    # Test cases: (batch_size, height, width, description)
    test_cases = [
        (1, 512, 512, "Single sample"),
        (2, 512, 512, "Batch of 2"),
        (4, 512, 512, "Batch of 4"),
    ]
    
    for batch_size, height, width, description in test_cases:
        try:
            # Create synthetic input
            input_tensor = torch.randn(batch_size, 3, height, width)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            # Validate output shape
            expected_shape = (batch_size, NUM_CLASSES, height, width)
            if output.shape == expected_shape:
                logger.info(f"✅ {description}: {input_tensor.shape} → {output.shape}")
                tests_passed.append(f"shape_{batch_size}_{height}_{width}")
            else:
                logger.error(f"❌ {description}: Expected {expected_shape}, got {output.shape}")
            
        except Exception as e:
            logger.error(f"❌ {description} failed: {e}")
    
    return tests_passed

def test_output_value_ranges(model):
    """Test output value ranges and distributions."""
    logger.info("📊 Testing output value ranges...")
    
    model.eval()
    tests_passed = []
    
    try:
        # Create synthetic input
        input_tensor = torch.randn(2, 3, 512, 512)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Test 1: Output is not all zeros
        if not torch.all(output == 0):
            logger.info("✅ Output is not all zeros")
            tests_passed.append("non_zero_output")
        else:
            logger.error("❌ Output is all zeros")
        
        # Test 2: Output has reasonable range
        output_min = output.min().item()
        output_max = output.max().item()
        output_range = output_max - output_min
        
        if output_range > 0.1:  # Some variation expected
            logger.info(f"✅ Output range: [{output_min:.3f}, {output_max:.3f}] (range: {output_range:.3f})")
            tests_passed.append("reasonable_range")
        else:
            logger.error(f"❌ Output range too small: [{output_min:.3f}, {output_max:.3f}]")
        
        # Test 3: Check for NaN or Inf
        if torch.isfinite(output).all():
            logger.info("✅ No NaN or Inf values in output")
            tests_passed.append("finite_values")
        else:
            logger.error("❌ Found NaN or Inf values in output")
        
        # Test 4: Test softmax behavior
        softmax_output = torch.softmax(output, dim=1)
        channel_sums = softmax_output.sum(dim=1)
        
        if torch.allclose(channel_sums, torch.ones_like(channel_sums), atol=1e-5):
            logger.info("✅ Softmax probabilities sum to 1.0")
            tests_passed.append("softmax_probabilities")
        else:
            logger.error(f"❌ Softmax probabilities don't sum to 1.0: {channel_sums.mean():.6f}")
        
        # Test 5: Test argmax behavior
        argmax_output = torch.argmax(output, dim=1)
        unique_classes = torch.unique(argmax_output)
        
        logger.info(f"✅ Argmax produces classes: {unique_classes.tolist()}")
        
        if len(unique_classes) > 1:
            logger.info("✅ Multiple classes predicted (diversity)")
            tests_passed.append("class_diversity")
        else:
            logger.warning("⚠️ Only single class predicted (possible mode collapse)")
        
        return tests_passed, output
        
    except Exception as e:
        logger.error(f"❌ Output value testing failed: {e}")
        return [], None

def test_synthetic_learning_task(model):
    """Test model with trivial synthetic learning task."""
    logger.info("🎯 Testing synthetic learning task...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Create trivial synthetic task: predict simple patterns
    def create_synthetic_data(batch_size=4):
        """Create synthetic data with known patterns."""
        images = torch.randn(batch_size, 3, 512, 512).to(device)
        
        # Create simple target patterns
        targets = torch.zeros(batch_size, 512, 512, dtype=torch.long).to(device)
        
        # Pattern 1: Top half = class 1, bottom half = class 2
        for i in range(batch_size):
            if i % 2 == 0:
                targets[i, :256, :] = 1  # Top half
                targets[i, 256:, :] = 2  # Bottom half
            else:
                targets[i, :, :256] = 1  # Left half
                targets[i, :, 256:] = 2  # Right half
        
        return images, targets
    
    # Test forward pass with synthetic data
    try:
        images, targets = create_synthetic_data(2)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        
        logger.info(f"✅ Synthetic task loss: {loss.item():.4f}")
        
        # Calculate synthetic IoU
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate IoU for synthetic patterns
            ious = []
            for class_id in range(1, NUM_CLASSES):
                pred_mask = (predictions == class_id)
                target_mask = (targets == class_id)
                
                intersection = (pred_mask & target_mask).sum().item()
                union = (pred_mask | target_mask).sum().item()
                
                if union > 0:
                    iou = intersection / union
                    ious.append(iou)
                    logger.info(f"✅ Synthetic Class {class_id} IoU: {iou:.3f}")
            
            mean_iou = np.mean(ious) if ious else 0.0
            logger.info(f"✅ Synthetic Mean IoU: {mean_iou:.3f}")
        
        # Test gradient computation
        loss.backward()
        
        # Check gradients
        has_gradients = False
        zero_gradients = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                total_params += 1
                if torch.all(param.grad == 0):
                    zero_gradients += 1
        
        if has_gradients:
            logger.info(f"✅ Gradients computed: {total_params - zero_gradients}/{total_params} non-zero")
        else:
            logger.error("❌ No gradients computed")
        
        tests_passed = []
        if loss.item() < 10.0:  # Reasonable loss range
            tests_passed.append("reasonable_loss")
        if has_gradients:
            tests_passed.append("gradients_computed")
        if zero_gradients < total_params * 0.5:  # Less than 50% zero gradients
            tests_passed.append("non_zero_gradients")
        
        return tests_passed, loss.item(), mean_iou
        
    except Exception as e:
        logger.error(f"❌ Synthetic learning task failed: {e}")
        import traceback
        traceback.print_exc()
        return [], float('inf'), 0.0

def test_weight_initialization(model):
    """Test weight initialization patterns."""
    logger.info("⚖️ Testing weight initialization...")
    
    tests_passed = []
    
    try:
        # Check for reasonable weight distributions
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.numel() > 1:
                mean = param.data.mean().item()
                std = param.data.std().item()
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                
                weight_stats[name] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'shape': list(param.shape)
                }
                
                logger.info(f"✅ {name}: mean={mean:.3f}, std={std:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
        
        # Check for reasonable initialization
        reasonable_init = True
        for name, stats in weight_stats.items():
            # Check for extremely small or large values
            if abs(stats['mean']) > 1.0 or stats['std'] > 2.0:
                logger.warning(f"⚠️ {name}: Unusual initialization (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
                reasonable_init = False
            
            # Check for all zeros
            if stats['std'] == 0:
                logger.error(f"❌ {name}: All weights are identical (std=0)")
                reasonable_init = False
        
        if reasonable_init:
            tests_passed.append("reasonable_initialization")
        
        return tests_passed, weight_stats
        
    except Exception as e:
        logger.error(f"❌ Weight initialization testing failed: {e}")
        return [], {}

def generate_validation_report(test_results):
    """Generate comprehensive model validation report."""
    report = {
        "test_type": "model_forward_pass_validation",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "total_tests": len(test_results),
        "passed_tests": sum(1 for result in test_results.values() if result.get('passed', False)),
        "test_results": test_results,
        "critical_issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Analyze critical issues
    if not test_results.get('model_creation', {}).get('passed', False):
        report["critical_issues"].append("Model creation failed")
    
    if not test_results.get('forward_pass', {}).get('passed', False):
        report["critical_issues"].append("Forward pass failed")
    
    if not test_results.get('synthetic_task', {}).get('passed', False):
        report["critical_issues"].append("Cannot perform basic learning")
    
    # Generate recommendations
    if not report["critical_issues"]:
        report["recommendations"].append("Model architecture validation passed - investigate loss function")
    else:
        report["recommendations"].append("Fix critical model architecture issues")
    
    # Save report
    report_path = Path("work_dirs/model_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Model validation report saved: {report_path}")
    return report

def main():
    """Run Phase 3.1: Model forward pass validation."""
    logger.info("🚀 PHASE 3.1: MODEL FORWARD PASS VALIDATION")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Model Creation
        logger.info("\n🏗️ TESTING MODEL CREATION")
        logger.info("-" * 40)
        model, creation_tests = test_model_creation()
        test_results['model_creation'] = {
            'passed': model is not None,
            'tests_passed': creation_tests,
            'details': f"Created model with {creation_tests.count('✅') if creation_tests else 0} validations"
        }
        
        if model is None:
            logger.error("💥 Model creation failed - cannot continue")
            return False
        
        # Test 2: Forward Pass Shapes
        logger.info("\n🔍 TESTING FORWARD PASS SHAPES")
        logger.info("-" * 40)
        shape_tests = test_forward_pass_shapes(model)
        test_results['forward_pass'] = {
            'passed': len(shape_tests) > 0,
            'tests_passed': shape_tests,
            'details': f"Passed {len(shape_tests)} shape tests"
        }
        
        # Test 3: Output Value Ranges
        logger.info("\n📊 TESTING OUTPUT VALUE RANGES")
        logger.info("-" * 40)
        value_tests, sample_output = test_output_value_ranges(model)
        test_results['output_values'] = {
            'passed': len(value_tests) > 0,
            'tests_passed': value_tests,
            'details': f"Passed {len(value_tests)} value tests"
        }
        
        # Test 4: Weight Initialization
        logger.info("\n⚖️ TESTING WEIGHT INITIALIZATION")
        logger.info("-" * 40)
        init_tests, weight_stats = test_weight_initialization(model)
        test_results['weight_initialization'] = {
            'passed': len(init_tests) > 0,
            'tests_passed': init_tests,
            'details': f"Checked {len(weight_stats)} weight tensors"
        }
        
        # Test 5: Synthetic Learning Task
        logger.info("\n🎯 TESTING SYNTHETIC LEARNING TASK")
        logger.info("-" * 40)
        synthetic_tests, loss_value, synthetic_iou = test_synthetic_learning_task(model)
        test_results['synthetic_task'] = {
            'passed': len(synthetic_tests) > 0,
            'tests_passed': synthetic_tests,
            'loss': loss_value,
            'synthetic_iou': synthetic_iou,
            'details': f"Loss: {loss_value:.4f}, IoU: {synthetic_iou:.3f}"
        }
        
        # Generate comprehensive report
        report = generate_validation_report(test_results)
        
        # Final Assessment
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        total_tests = sum(len(result.get('tests_passed', [])) for result in test_results.values())
        logger.info(f"Total tests run: {total_tests}")
        
        if report["critical_issues"]:
            logger.error("❌ CRITICAL ISSUES FOUND:")
            for issue in report["critical_issues"]:
                logger.error(f"  • {issue}")
            logger.error("🚫 MODEL ARCHITECTURE HAS FUNDAMENTAL PROBLEMS")
            return False
        else:
            logger.info("✅ NO CRITICAL ARCHITECTURE ISSUES FOUND")
            logger.info("🎯 MODEL ARCHITECTURE IS FUNCTIONAL")
            logger.info("📋 Next Step: Investigate loss function and optimization")
            
        return True
        
    except Exception as e:
        logger.error(f"💥 Model validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)