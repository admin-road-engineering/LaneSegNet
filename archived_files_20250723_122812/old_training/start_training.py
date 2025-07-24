#!/usr/bin/env python3
"""
Phase 3.2: Training Launcher for AEL Lane Detection
Starts MMSegmentation training with monitoring
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import mmseg
        import torch
        print(f"✓ MMSegmentation: {mmseg.__version__}")
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU Count: {torch.cuda.device_count()}")
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def validate_data():
    """Validate training data is properly set up."""
    data_root = Path("data/ael_mmseg")
    
    # Check required directories
    required_dirs = [
        "img_dir/train",
        "ann_dir/train", 
        "img_dir/val",
        "ann_dir/val"
    ]
    
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
            return False
        
        files = list(full_path.glob("*"))
        print(f"✓ {dir_path}: {len(files)} files")
    
    return True

def run_training():
    """Run the training process."""
    config_file = "configs/ael_swin_upernet_training.py"
    work_dir = "work_dirs/ael_training"
    
    # Ensure work directory exists
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    # Training command
    cmd = [
        sys.executable, "-m", "mmseg.apis.train",
        config_file,
        "--work-dir", work_dir,
        "--seed", "42",
        "--deterministic"
    ]
    
    print(f"Starting training with command:")
    print(" ".join(cmd))
    print(f"Working directory: {work_dir}")
    print(f"Config file: {config_file}")
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    try:
        # Start training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Monitor output
        print("\n" + "="*60)
        print("TRAINING OUTPUT")
        print("="*60)
        
        for line in process.stdout:
            print(line.strip())
            
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function."""
    print("Phase 3.2: Starting AEL Lane Detection Training")
    print("Target: 80-85% mIoU with 4-class lane detection")
    print("="*60)
    
    # Check dependencies
    print("\n1. Checking Dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    # Validate data
    print("\n2. Validating Training Data...")
    if not validate_data():
        print("❌ Data validation failed")
        return 1
    
    # Start training
    print("\n3. Starting Training...")
    start_time = time.time()
    
    success = run_training()
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    
    print("\n" + "="*60)
    if success:
        print("✓ Training completed successfully!")
        print(f"Total time: {int(hours)}h {int(minutes)}m")
        print("Check work_dirs/ael_training/ for results")
    else:
        print("❌ Training failed")
        print(f"Time elapsed: {int(hours)}h {int(minutes)}m")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())