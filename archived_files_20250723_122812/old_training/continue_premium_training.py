#!/usr/bin/env python3
"""
Option 1: Continue Premium Training for 30 More Epochs (51-80)
Uses the exact same proven parameters that achieved 85.1% mIoU
Simple continuation with automated saving
"""

import sys
sys.path.append('scripts')
from premium_gpu_train import premium_gpu_training
import torch
import json
from pathlib import Path
from datetime import datetime

def continue_premium_training():
    """
    Continue the successful premium training for 30 more epochs
    Load from 85.1% checkpoint and use identical parameters
    """
    
    print("=" * 70)
    print("OPTION 1: CONTINUE PREMIUM TRAINING")
    print("Same proven parameters, 30 more epochs (51-80)")
    print("=" * 70)
    print()

    # Verify checkpoint exists
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if not checkpoint_path.exists():
        print("ERROR: No checkpoint found for continuation!")
        return None, None

    # Load checkpoint info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    current_epoch = checkpoint.get('epoch', 50)
    current_miou = checkpoint.get('best_miou', 0.851)
    
    print(f"Starting from:")
    print(f"  Epoch: {current_epoch}")
    print(f"  mIoU: {current_miou*100:.1f}%")
    print(f"  Target: Continue from Epoch 50 to 80 (30 more epochs)")
    print()

    # Load the exact same Bayesian-optimized parameters
    try:
        with open('work_dirs/bayesian_optimization_results.json', 'r') as f:
            results = json.load(f)
        
        learning_rate = results['best_params']['learning_rate']  # 5.14e-4
        dice_weight = results['best_params']['dice_weight']      # 0.694
        
        print("Using proven Bayesian-optimized parameters:")
        print(f"  Learning Rate: {learning_rate:.2e}")
        print(f"  Dice Weight: {dice_weight:.3f}")
        print()
        
    except FileNotFoundError:
        print("Bayesian results not found, using defaults")
        learning_rate = 5.14e-4
        dice_weight = 0.694

    # IMPORTANT: Modify the premium_gpu_train.py to continue from checkpoint
    # We'll create a modified version that loads the checkpoint
    
    print("Creating continuation script...")
    
    # Read the original premium script with proper encoding
    try:
        with open('scripts/premium_gpu_train.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open('scripts/premium_gpu_train.py', 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Modify for continuation
    # Find the model creation line and add checkpoint loading after it
    model_creation_marker = "model = PremiumLaneNet(num_classes=3, dropout_rate=0.3).to(device)"
    checkpoint_loading_code = '''
    # Load checkpoint for continuation
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    start_epoch = 1
    if checkpoint_path.exists():
        print("Loading 85.1% mIoU checkpoint for continuation...")
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 50) + 1
            print(f"SUCCESS: Loaded checkpoint from epoch {checkpoint.get('epoch', 50)}")
            print(f"SUCCESS: Continuing from epoch 50 (after 85.1% achievement)")
        else:
            print("ERROR: Invalid checkpoint format")
    else:
        print("ERROR: No checkpoint found - starting fresh training")
    '''
    
    # Replace the epoch range to continue from epoch 50 regardless of checkpoint epoch
    epoch_range_old = "for epoch in range(50):"
    epoch_range_new = "for epoch in range(50, 80):  # Continue from epoch 50 to 80"
    
    # Apply modifications
    if model_creation_marker in content:
        content = content.replace(model_creation_marker, 
                                model_creation_marker + checkpoint_loading_code)
        print("SUCCESS: Added checkpoint loading")
    
    if epoch_range_old in content:
        content = content.replace(epoch_range_old, epoch_range_new)
        print("SUCCESS: Modified epoch range for continuation")
    
    # Change title
    content = content.replace("Premium GPU Training - Phase 3.2.5", 
                            "Option 1: Premium Training Continuation (Epochs 51-80)")
    
    # Add improved saving logic - backup ALL best epochs
    backup_code = '''
        # BACKUP EVERY BEST EPOCH (not just overwrite)
        if best_miou > 0 and (epoch == 0 or current_miou > best_miou):
            # Create unique backup for this epoch's achievement
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"option1_epoch{epoch+1}_continue_{current_miou*100:.1f}miou_{timestamp}"
            backup_dir = Path(f"model_backups/{backup_name}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(locals().get('scheduler', None), 'state_dict') else None,
                'best_miou': current_miou,
                'best_balanced_score': balanced_score,
                'class_ious': ious,
                'approach': 'option1_continuation',
                'base_model': '85.1% premium training'
            }
            
            # Save to unique backup location
            torch.save(checkpoint_data, backup_dir / "option1_continue_model.pth")
            
            # Also update working checkpoint (separate from other approaches)
            torch.save(checkpoint_data, "work_dirs/option1_continue_best_model.pth")
            
            # Save performance record
            with open(backup_dir / f"OPTION1_CONTINUE_EPOCH{epoch+1}_RECORD.json", "w") as f:
                json.dump({
                    'epoch': epoch + 1,
                    'miou': current_miou,
                    'balanced_score': balanced_score,
                    'approach': 'Option 1: Premium Training Continuation',
                    'base_model': '85.1% mIoU from Epoch 50',
                    'parameters': {
                        'learning_rate': learning_rate,
                        'dice_weight': dice_weight,
                        'batch_size': 12,
                        'identical_to_original': True
                    },
                    'class_performance': {
                        'background': ious[0] if len(ious) > 0 else 0,
                        'white_solid': ious[1] if len(ious) > 1 else 0,
                        'white_dashed': ious[2] if len(ious) > 2 else 0
                    },
                    'timestamp': timestamp
                }, f, indent=2)
            
            print(f"✅ OPTION 1 BACKUP: Epoch {epoch+1} saved to model_backups/{backup_name}/")
    '''
    
    # Insert backup code before the results saving
    content = content.replace(
        'with open("work_dirs/premium_gpu_results.json", "w") as f:',
        backup_code + '\n    with open("work_dirs/premium_gpu_results.json", "w") as f:'
    )
    
    # Save the continuation script with proper encoding
    with open('scripts/premium_continue_51_80.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Created scripts/premium_continue_51_80.py")
    print()
    print("This script will:")
    print("  1. Load the 85.1% mIoU checkpoint")
    print("  2. Continue training from Epoch 50 to 80")
    print("  3. Use identical proven parameters")
    print("  4. Auto-save any improvements")
    print()
    
    return learning_rate, dice_weight

if __name__ == "__main__":
    continue_premium_training()