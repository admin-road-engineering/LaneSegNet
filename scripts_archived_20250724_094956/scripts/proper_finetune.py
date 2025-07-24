#!/usr/bin/env python3
"""
PROPER Fine-Tuning Script
Load 85.1% mIoU checkpoint and continue with fine-tuning parameters
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime

def main():
    print("=" * 60)
    print("PROPER FINE-TUNING - Loading 85.1% mIoU Checkpoint")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Cannot proceed with fine-tuning!")
        return
    
    # Load and inspect checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} parameters")
        else:
            print(f"  {key}: {checkpoint[key]}")
    
    print(f"\nCheckpoint loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best mIoU in checkpoint: {checkpoint.get('best_miou', 'unknown'):.1f}%")
    
    # The issue is we need to modify the premium_gpu_train.py to:
    # 1. Load this checkpoint
    # 2. Continue from epoch 51
    # 3. Use learning rate 1e-5
    
    print("\n" + "=" * 60)
    print("SOLUTION: Modify premium_gpu_train.py directly")
    print("=" * 60)
    
    modifications_needed = """
    To fix the fine-tuning, we need to modify premium_gpu_train.py:
    
    1. Add checkpoint loading before training starts:
       ```python
       # Load checkpoint if exists
       if Path('work_dirs/premium_gpu_best_model.pth').exists():
           checkpoint = torch.load('work_dirs/premium_gpu_best_model.pth')
           model.load_state_dict(checkpoint['model_state_dict'])
           print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
           start_epoch = checkpoint['epoch'] + 1
       ```
    
    2. Change learning rate to fine-tuning value:
       Change: learning_rate = 5.14e-4
       To: learning_rate = 1e-5
    
    3. Start from epoch 51 instead of 1:
       Change: for epoch in range(1, 51):
       To: for epoch in range(start_epoch, start_epoch + 30):
    """
    
    print(modifications_needed)
    
    # Create a simple modification script
    print("\nCreating automatic fix...")
    
    with open('fix_premium_script.py', 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Fix premium_gpu_train.py for proper fine-tuning
"""

def fix_premium_script():
    print("Reading premium_gpu_train.py...")
    
    try:
        with open('scripts/premium_gpu_train.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open('scripts/premium_gpu_train.py', 'r', encoding='latin-1') as f:
            content = f.read()
    
    print("Original file loaded successfully")
    
    # Key modifications for fine-tuning
    modifications = []
    
    # 1. Change learning rate to fine-tuning value
    if 'learning_rate = 5.14e-4' in content:
        content = content.replace('learning_rate = 5.14e-4', 'learning_rate = 1e-5')
        modifications.append("✓ Changed learning rate to 1e-5")
    
    # 2. Add checkpoint loading (we'll add this manually)
    checkpoint_loading = """
    # Load checkpoint for fine-tuning
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    start_epoch = 1
    if checkpoint_path.exists():
        print("Loading 85.1% mIoU checkpoint for fine-tuning...")
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = 51  # Continue from epoch 51
            print(f"✅ Loaded checkpoint from epoch {{checkpoint.get('epoch', 50)}}")
            print(f"✅ Starting fine-tuning from epoch {{start_epoch}}")
        else:
            print("❌ Invalid checkpoint format")
    else:
        print("❌ No checkpoint found - starting fresh training")
    """
    
    # Save modified content
    with open('scripts/premium_gpu_train_finetune.py', 'w') as f:
        f.write(content)
    
    print("\\n✅ Created: scripts/premium_gpu_train_finetune.py")
    print("\\nNext steps:")
    print("1. Manually add checkpoint loading code after model creation")
    print("2. Change epoch range to use start_epoch")
    print("3. Run the modified script")

if __name__ == "__main__":
    fix_premium_script()
''')
    
    print("✅ Created fix_premium_script.py")
    print("\nRun: python fix_premium_script.py")
    print("Then manually add checkpoint loading code")

if __name__ == "__main__":
    main()