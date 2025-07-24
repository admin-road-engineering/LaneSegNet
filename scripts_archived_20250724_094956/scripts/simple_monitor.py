#!/usr/bin/env python3
"""
Simple Training Monitor (no external dependencies)
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress by checking files."""
    print("=== Phase 3.2 Training Monitor ===")
    print(f"Check time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    work_dir = Path("work_dirs")
    
    # Check baseline model
    baseline_model = work_dir / "best_model.pth"
    if baseline_model.exists():
        size_mb = baseline_model.stat().st_size / 1024**2
        print(f"Baseline Model: COMPLETED ({size_mb:.1f}MB)")
    else:
        print("Baseline Model: Not found")
    
    # Check enhanced model
    enhanced_model = work_dir / "enhanced_best_model.pth"
    if enhanced_model.exists():
        size_mb = enhanced_model.stat().st_size / 1024**2
        mod_time = datetime.fromtimestamp(enhanced_model.stat().st_mtime)
        time_since_update = datetime.now().timestamp() - enhanced_model.stat().st_mtime
        
        print(f"Enhanced Model: FOUND ({size_mb:.1f}MB)")
        print(f"  Last updated: {mod_time.strftime('%H:%M:%S')}")
        print(f"  Time since update: {time_since_update/60:.0f} minutes")
        
        if time_since_update < 600:  # 10 minutes
            print("  Status: Recently updated - training likely ACTIVE")
        else:
            print("  Status: No recent updates - likely COMPLETED")
    else:
        print("Enhanced Model: Not created yet - training IN PROGRESS")
    
    # Check training results
    results_file = work_dir / "enhanced_training_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print()
            print("FINAL RESULTS:")
            print(f"  Best mIoU: {results.get('best_miou', 0):.1%}")
            print(f"  Epochs: {results.get('num_epochs_completed', 'Unknown')}")
            print(f"  Training Time: {results.get('training_time_hours', 0):.1f}h")
            
            target_80 = results.get('target_80_achieved', False)
            target_85 = results.get('target_85_achieved', False)
            
            if target_85:
                print("  STATUS: EXCELLENT! 85%+ target achieved!")
            elif target_80:
                print("  STATUS: SUCCESS! 80-85% target achieved!")
            else:
                print("  STATUS: In progress or below target")
                
        except Exception as e:
            print(f"Results file error: {e}")
    else:
        print("Training Results: Not available yet")
    
    # Check GPU usage (simple)
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 1.0:
                print(f"GPU: Active ({allocated:.1f}GB allocated)")
            else:
                print("GPU: Idle")
        else:
            print("GPU: Not available")
    except:
        print("GPU: Cannot check")
    
    print()
    
    # Overall status
    if results_file.exists():
        print("OVERALL STATUS: Training COMPLETED")
        print("Next: Run final evaluation and integration")
    elif enhanced_model.exists():
        time_diff = datetime.now().timestamp() - enhanced_model.stat().st_mtime
        if time_diff < 600:  # 10 minutes
            print("OVERALL STATUS: Training IN PROGRESS")
            print("Check again in 15-20 minutes")
        else:
            print("OVERALL STATUS: Likely COMPLETED")
            print("May need to check if process finished")
    else:
        print("OVERALL STATUS: Training IN PROGRESS")
        print("Enhanced model not created yet")

if __name__ == "__main__":
    monitor_training()