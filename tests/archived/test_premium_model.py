#!/usr/bin/env python3
"""
Quick test to verify premium model architecture works
"""

import torch
import sys
import os

# Add scripts directory to path
sys.path.append('scripts')

from premium_gpu_train import PremiumLaneNet

def test_premium_model():
    """Test premium model forward pass."""
    print("Testing Premium Model Architecture...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, testing on CPU")
        device = torch.device("cpu")
    else:
        print(f"Testing on GPU: {torch.cuda.get_device_name()}")
        device = torch.device("cuda")
    
    # Initialize model
    model = PremiumLaneNet(num_classes=4, dropout_rate=0.3).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test with sample input
    batch_size = 2
    sample_input = torch.randn(batch_size, 3, 512, 512).to(device)
    
    print(f"Input shape: {sample_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(sample_input)
        
        print(f"Output shape: {output.shape}")
        print("SUCCESS: MODEL ARCHITECTURE TEST PASSED!")
        print()
        print("Expected output shape: (2, 4, 512, 512)")
        print(f"Actual output shape:   {output.shape}")
        
        if output.shape == (batch_size, 4, 512, 512):
            print("PERFECT: Model ready for training.")
            return True
        else:
            print("ERROR: Shape mismatch - architecture needs adjustment")
            return False
            
    except Exception as e:
        print(f"ERROR: MODEL TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_premium_model()
    
    if success:
        print("\nREADY: Premium model ready! You can now run:")
        print("   run_premium_gpu.bat")
    else:
        print("\nDEBUG: Model needs debugging before training")