#!/usr/bin/env python3
"""
Training Progress Monitor for Phase 3.2
Checks training status and provides updates without terminal access
"""

import json
import time
from pathlib import Path
import os
import psutil
from datetime import datetime

def check_training_status():
    """Check current training status and progress."""
    print("=== Phase 3.2 Training Monitor ===")
    print(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if training process is running
    training_active = False
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if ('python' in proc.info['name'].lower() and 
                    proc.info['cmdline'] and 
                    any('swin_transformer_train.py' in str(cmd) for cmd in proc.info['cmdline'])):
                    
                    training_active = True
                    print("TRAINING STATUS: ACTIVE")
                    print(f"  Process ID: {proc.info['pid']}")
                    print(f"  CPU Usage: {proc.cpu_percent():.1f}%")
                    print(f"  Memory: {proc.info['memory_info'].rss / 1024**2:.0f}MB")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        print("TRAINING STATUS: Cannot check process (psutil not available)")
        print("Checking file indicators instead...")
    
    if not training_active:
        print("TRAINING STATUS: NOT DETECTED or COMPLETED")
    
    print()
    
    # Check training outputs
    work_dir = Path("work_dirs")
    if not work_dir.exists():
        print("WORK DIRECTORY: Not found")
        return
    
    print("TRAINING ARTIFACTS:")
    
    # Check baseline model
    baseline_model = work_dir / "best_model.pth"
    if baseline_model.exists():
        size_mb = baseline_model.stat().st_size / 1024**2
        mod_time = datetime.fromtimestamp(baseline_model.stat().st_mtime)
        print(f"  Baseline Model: COMPLETED ({size_mb:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
    else:
        print("  Baseline Model: Not found")
    
    # Check enhanced model
    enhanced_model = work_dir / "enhanced_best_model.pth"
    if enhanced_model.exists():
        size_mb = enhanced_model.stat().st_size / 1024**2
        mod_time = datetime.fromtimestamp(enhanced_model.stat().st_mtime)
        print(f"  Enhanced Model: FOUND ({size_mb:.1f}MB, last updated: {mod_time.strftime('%H:%M:%S')})")
        
        # Check if recently updated (within last 5 minutes)
        time_diff = datetime.now().timestamp() - enhanced_model.stat().st_mtime
        if time_diff < 300:  # 5 minutes
            print("    Status: Recently updated - training likely active")
        else:
            print("    Status: No recent updates - may be complete")
    else:
        print("  Enhanced Model: Not found yet")
    
    # Check training results
    results_file = work_dir / "enhanced_training_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print()
            print("TRAINING RESULTS:")
            print(f"  Best mIoU: {results.get('best_miou', 0):.1%}")
            print(f"  Epochs Completed: {results.get('num_epochs_completed', 'Unknown')}")
            print(f"  Training Time: {results.get('training_time_hours', 0):.1f} hours")
            print(f"  80% Target: {'ACHIEVED' if results.get('target_80_achieved') else 'Not yet'}")
            print(f"  85% Target: {'ACHIEVED' if results.get('target_85_achieved') else 'Not yet'}")
            
        except Exception as e:
            print(f"  Results file exists but couldn't read: {e}")
    else:
        print("  Training Results: Not available yet")
    
    print()
    
    # Provide status summary
    if enhanced_model.exists() and results_file.exists():
        print("OVERALL STATUS: Training appears COMPLETED")
        print("  Next step: Run final evaluation and integration")
    elif enhanced_model.exists():
        time_diff = datetime.now().timestamp() - enhanced_model.stat().st_mtime
        if time_diff < 300:
            print("OVERALL STATUS: Training IN PROGRESS")
            print("  Model is being updated, check back in 10-15 minutes")
        else:
            print("OVERALL STATUS: Training may be COMPLETED")
            print("  Run evaluation to confirm final results")
    else:
        print("OVERALL STATUS: Training IN PROGRESS or not started")
        print("  Enhanced model not created yet")

def estimate_completion_time():
    """Estimate when training might complete."""
    print()
    print("ESTIMATED COMPLETION:")
    
    enhanced_model = Path("work_dirs/enhanced_best_model.pth")
    
    if enhanced_model.exists():
        # Check how long since model was created
        created_time = datetime.fromtimestamp(enhanced_model.stat().st_ctime)
        now = datetime.now()
        elapsed = now - created_time
        
        print(f"  Training started: {created_time.strftime('%H:%M:%S')}")
        print(f"  Elapsed time: {elapsed.total_seconds() / 3600:.1f} hours")
        
        # Enhanced training typically takes 2-4 hours for 40 epochs
        estimated_total_hours = 3.0  # Conservative estimate
        
        if elapsed.total_seconds() / 3600 >= estimated_total_hours:
            print("  Estimated status: Should be complete soon")
        else:
            remaining_hours = estimated_total_hours - (elapsed.total_seconds() / 3600)
            completion_time = now + time.timedelta(hours=remaining_hours)
            print(f"  Estimated completion: {completion_time.strftime('%H:%M:%S')} ({remaining_hours:.1f}h remaining)")
    else:
        print("  Cannot estimate - enhanced training not started")

def check_gpu_usage():
    """Check GPU memory usage as indicator of training activity."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print()
            print("GPU STATUS:")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
            
            if allocated > 1.0:
                print("  Status: GPU actively in use (likely training)")
            else:
                print("  Status: GPU mostly idle")
        else:
            print()
            print("GPU STATUS: CUDA not available")
    except ImportError:
        print()
        print("GPU STATUS: Cannot check (PyTorch not available)")

def main():
    """Main monitoring function."""
    check_training_status()
    estimate_completion_time()
    check_gpu_usage()
    
    print()
    print("=== NEXT STEPS ===")
    print("1. Run this monitor again in 10-15 minutes to check progress")
    print("2. When training is complete, run: python scripts/evaluate_final.py")
    print("3. Then proceed with API integration")
    print()
    print("To run this monitor again:")
    print("  python scripts/monitor_training.py")

if __name__ == "__main__":
    main()