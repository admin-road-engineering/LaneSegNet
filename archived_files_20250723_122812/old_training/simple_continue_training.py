#!/usr/bin/env python3
"""
Simple Option 1: Continue training from 85.1% checkpoint
Uses the same parameters that achieved the breakthrough
"""

import sys
sys.path.append('scripts')
import torch
from pathlib import Path
from datetime import datetime
import json

# Import the same function that achieved 85.1%
from premium_gpu_train import premium_gpu_training

def simple_continue_from_checkpoint():
    """
    Load 85.1% checkpoint and continue with identical parameters
    """
    print("=" * 70)
    print("OPTION 1: SIMPLE PREMIUM CONTINUATION")  
    print("Continue from 85.1% using identical proven parameters")
    print("=" * 70)
    print()

    # Verify checkpoint exists
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if not checkpoint_path.exists():
        print("ERROR: No 85.1% checkpoint found!")
        return None, None

    # Load checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    current_epoch = checkpoint.get('epoch', 49)
    current_miou = checkpoint.get('best_miou', 0.851)
    
    print(f"CONTINUATION STARTING POINT:")
    print(f"  Current Epoch: {current_epoch}")
    print(f"  Current mIoU: {current_miou*100:.1f}%")
    print(f"  Approach: Identical parameters that achieved this performance")
    print()

    # Use the exact same Bayesian-optimized parameters
    try:
        with open('work_dirs/bayesian_optimization_results.json', 'r') as f:
            results = json.load(f)
        
        learning_rate = results['best_params']['learning_rate']  # 5.14e-4
        dice_weight = results['best_params']['dice_weight']      # 0.694
        
        print("USING PROVEN BAYESIAN-OPTIMIZED PARAMETERS:")
        print(f"  Learning Rate: {learning_rate:.2e}")
        print(f"  Dice Weight: {dice_weight:.3f}")
        print("  All other parameters: Identical to original premium training")
        print()
        
    except FileNotFoundError:
        print("Bayesian results not found, using known values")
        learning_rate = 5.14e-4
        dice_weight = 0.694

    print("STARTING CONTINUATION...")
    print("This will load the 85.1% model and continue training")
    print("Expected improvements: Gradual increase to 86-87% mIoU")
    print()
    
    # Use the same premium_gpu_training function with identical parameters
    # The function will automatically load the checkpoint if it exists
    final_miou, balanced_score = premium_gpu_training(
        learning_rate=learning_rate,
        dice_weight=dice_weight
    )
    
    print("=" * 70)
    print("OPTION 1 CONTINUATION COMPLETE!")
    print(f"Final mIoU: {final_miou*100:.1f}%" if final_miou else "Training interrupted")
    print("=" * 70)
    
    return final_miou, balanced_score

if __name__ == "__main__":
    simple_continue_from_checkpoint()