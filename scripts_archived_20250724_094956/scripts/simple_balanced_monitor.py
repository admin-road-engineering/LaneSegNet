#!/usr/bin/env python3
"""
Simple monitor for Phase 3.2.5 balanced training (no Unicode)
"""

import json
import time
from pathlib import Path
from datetime import datetime

def simple_monitor():
    """Simple monitoring without Unicode characters."""
    print("=== Phase 3.2.5: Balanced Training Monitor ===")
    print(f"Check time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    work_dir = Path("work_dirs")
    
    # Check all models
    print("MODEL STATUS:")
    
    # Baseline
    baseline = work_dir / "best_model.pth"
    if baseline.exists():
        size = baseline.stat().st_size / 1024**2
        print(f"  Baseline:  {size:.1f}MB - 52.0% mIoU")
    
    # Enhanced (problematic)
    enhanced = work_dir / "enhanced_best_model.pth"
    if enhanced.exists():
        size = enhanced.stat().st_size / 1024**2
        print(f"  Enhanced:  {size:.1f}MB - 48.8% mIoU (class imbalance issue)")
    
    # Balanced (current)
    balanced = work_dir / "balanced_best_model.pth"
    if balanced.exists():
        size = balanced.stat().st_size / 1024**2
        mod_time = datetime.fromtimestamp(balanced.stat().st_mtime)
        minutes_ago = (datetime.now().timestamp() - balanced.stat().st_mtime) / 60
        
        print(f"  Balanced:  {size:.1f}MB - DiceFocal Loss Fix")
        print(f"             Last update: {mod_time.strftime('%H:%M:%S')} ({minutes_ago:.0f} min ago)")
        
        if minutes_ago < 10:
            print("             Status: TRAINING ACTIVE")
        else:
            print("             Status: TRAINING COMPLETED")
        
        # Size check
        if size <= 20:
            print("             Size: OPTIMIZED")
        else:
            print("             Size: ACCEPTABLE")
    else:
        print("  Balanced:  NOT FOUND - Training in progress")
    
    print()
    
    # Check results
    results_file = work_dir / "balanced_training_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print("TRAINING RESULTS:")
            miou = results.get('best_miou', 0)
            balanced_score = results.get('best_balanced_score', 0)
            hours = results.get('training_time_hours', 0)
            epochs = results.get('num_epochs_completed', 0)
            
            print(f"  Overall mIoU:     {miou:.1%}")
            print(f"  Balanced Score:   {balanced_score:.1%}")
            print(f"  Training Time:    {hours:.1f} hours")
            print(f"  Epochs:           {epochs}")
            
            # Per-class results
            class_ious = results.get('final_per_class_ious', [])
            class_names = results.get('class_names', [])
            
            if class_ious and len(class_ious) == 4:
                print()
                print("  Per-Class Performance:")
                for name, iou in zip(class_names, class_ious):
                    status = "GOOD" if iou > 0.5 else ("OK" if iou > 0.2 else "POOR")
                    print(f"    {status:4} {name:15}: {iou:.1%}")
                
                # Class imbalance check
                lane_ious = class_ious[1:]  # Exclude background
                min_lane = min(lane_ious) if lane_ious else 0
                
                if min_lane > 0.5:
                    print("  Class Imbalance:  FIXED (all lanes >50%)")
                elif min_lane > 0.2:
                    print(f"  Class Imbalance:  IMPROVING (min {min_lane:.1%})")
                else:
                    print(f"  Class Imbalance:  PERSISTS (min {min_lane:.1%})")
            
            print()
            
            # Target assessment
            if miou >= 0.85:
                print("TARGET STATUS: OUTSTANDING! 85%+ achieved")
                status = "EXCELLENT"
            elif miou >= 0.80:
                print("TARGET STATUS: SUCCESS! 80-85% achieved")
                status = "SUCCESS"
            elif miou >= 0.70:
                print("TARGET STATUS: GOOD! 70%+ achieved")
                status = "GOOD"
            elif miou > 0.60:
                print("TARGET STATUS: IMPROVED over baseline")
                status = "IMPROVED"
            else:
                print("TARGET STATUS: NEEDS ANALYSIS")
                status = "ANALYSIS"
            
            print()
            
            # Comparison
            print("MODEL COMPARISON:")
            baseline_improvement = (miou - 0.52) * 100
            enhanced_improvement = (miou - 0.488) * 100
            
            print(f"  vs Baseline (52%):   {baseline_improvement:+.1f} points")
            print(f"  vs Enhanced (48.8%): {enhanced_improvement:+.1f} points")
            
            return status, miou
            
        except Exception as e:
            print(f"Error reading results: {e}")
            return "ERROR", 0
    else:
        print("TRAINING RESULTS: Not available yet")
        return "IN_PROGRESS", 0

def check_training_active():
    """Check if training is currently active."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            
            print("GPU STATUS:")
            print(f"  Memory allocated: {allocated:.1f}GB")
            
            if allocated > 2.0:
                print("  Status: TRAINING ACTIVE")
                return True
            elif allocated > 0.5:
                print("  Status: MODEL LOADED")
                return False
            else:
                print("  Status: IDLE")
                return False
        else:
            print("GPU: Not available")
            return False
    except:
        print("GPU: Cannot check")
        return False

def main():
    """Main function."""
    status, miou = simple_monitor()
    
    print()
    is_training = check_training_active()
    
    print()
    print("NEXT STEPS:")
    
    if status == "EXCELLENT" or status == "SUCCESS":
        print("1. Training target achieved!")
        print("2. Run final evaluation")
        print("3. Deploy to production")
    elif status == "GOOD" or status == "IMPROVED":
        print("1. Significant improvement achieved")
        print("2. Run comprehensive evaluation")
        print("3. Consider production deployment")
    elif status == "IN_PROGRESS":
        if is_training:
            print("1. Training is active - wait for completion")
            print("2. Check again in 30-60 minutes")
        else:
            print("1. Training may have completed")
            print("2. Check if results file will be updated")
    else:
        print("1. Analyze training results")
        print("2. Consider parameter adjustments")
    
    print()
    print("COMMANDS:")
    print("  python scripts/simple_balanced_monitor.py  # Check progress")
    print("  python scripts/balanced_eval.py           # Run evaluation")

if __name__ == "__main__":
    main()