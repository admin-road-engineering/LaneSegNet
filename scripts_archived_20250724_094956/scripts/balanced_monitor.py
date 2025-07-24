#!/usr/bin/env python3
"""
Monitor Phase 3.2.5 balanced training progress
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_balanced_training():
    """Monitor balanced training with DiceFocal loss."""
    print("=== Phase 3.2.5: Balanced Training Monitor ===")
    print(f"Check time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    work_dir = Path("work_dirs")
    
    # Check previous models
    print("Previous Models:")
    
    baseline_model = work_dir / "best_model.pth"
    if baseline_model.exists():
        size_mb = baseline_model.stat().st_size / 1024**2
        print(f"  Baseline (Simple CNN): {size_mb:.1f}MB - 52% mIoU")
    
    enhanced_model = work_dir / "enhanced_best_model.pth"
    if enhanced_model.exists():
        size_mb = enhanced_model.stat().st_size / 1024**2
        print(f"  Enhanced (Deep CNN): {size_mb:.1f}MB - 48.8% mIoU (class imbalance)")
    
    print()
    
    # Check balanced model (Phase 3.2.5)
    balanced_model = work_dir / "balanced_best_model.pth"
    if balanced_model.exists():
        size_mb = balanced_model.stat().st_size / 1024**2
        mod_time = datetime.fromtimestamp(balanced_model.stat().st_mtime)
        time_since_update = datetime.now().timestamp() - balanced_model.stat().st_mtime
        
        print(f"SUCCESS: BALANCED MODEL FOUND: {size_mb:.1f}MB")
        print(f"   Created: {mod_time.strftime('%H:%M:%S')}")
        print(f"   Time since update: {time_since_update/60:.0f} minutes")
        
        if time_since_update < 600:  # 10 minutes
            print("   Status: Recently updated - training likely ACTIVE")
        else:
            print("   Status: Training likely COMPLETED")
        
        # Model size assessment
        if size_mb <= 20:
            print(f"   Size Assessment: OPTIMIZED (target: 10-20MB)")
        elif size_mb <= 30:
            print(f"   Size Assessment: ACCEPTABLE (slightly above target)")
        else:
            print(f"   Size Assessment: TOO LARGE (may overfit)")
    else:
        print("WAITING: Balanced Model: Not created yet - training IN PROGRESS or not started")
    
    print()
    
    # Check training results
    results_file = work_dir / "balanced_training_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print("🎯 BALANCED TRAINING RESULTS:")
            overall_miou = results.get('best_miou', 0)
            balanced_score = results.get('best_balanced_score', 0)
            
            print(f"   Overall mIoU: {overall_miou:.1%}")
            print(f"   Balanced Score: {balanced_score:.1%}")
            print(f"   Training Time: {results.get('training_time_hours', 0):.1f} hours")
            print(f"   Epochs: {results.get('num_epochs_completed', 'Unknown')}")
            print(f"   Architecture: {results.get('architecture', 'OptimizedLaneNet')}")
            
            # Per-class results
            class_ious = results.get('final_per_class_ious', [])
            class_names = results.get('class_names', ['background', 'white_solid', 'white_dashed', 'yellow_solid'])
            
            if class_ious:
                print()
                print("   Per-class Performance:")
                for name, iou in zip(class_names, class_ious):
                    status = "✅" if iou > 0.5 else ("⚠" if iou > 0.2 else "❌")
                    print(f"     {status} {name}: {iou:.1%}")
                
                # Check class imbalance fix
                lane_classes = class_ious[1:]  # Exclude background
                min_lane_iou = min(lane_classes) if lane_classes else 0
                
                if min_lane_iou > 0.5:
                    print(f"   🎯 CLASS IMBALANCE FIXED: All lane classes >50%")
                elif min_lane_iou > 0.2:
                    print(f"   📈 IMPROVING: Min lane IoU {min_lane_iou:.1%}")
                else:
                    print(f"   ⚠ STILL IMBALANCED: Min lane IoU {min_lane_iou:.1%}")
            
            print()
            
            # Target assessment
            target_70 = results.get('target_70_achieved', False)
            target_80 = results.get('target_80_achieved', False)
            target_85 = results.get('target_85_achieved', False)
            class_balance_fixed = results.get('class_imbalance_fixed', False)
            
            if target_85:
                print("🏆 OUTSTANDING SUCCESS! 85%+ target achieved!")
                print("   Ready for production deployment")
                status = "EXCELLENT"
            elif target_80:
                print("🎯 SUCCESS! 80%+ target achieved!")
                print("   Ready for production deployment")
                status = "SUCCESS" 
            elif target_70:
                print("✅ GOOD PROGRESS! 70%+ target achieved!")
                print("   Significant improvement - consider production")
                status = "GOOD"
            elif overall_miou > 0.60:
                print("📈 IMPROVED! Above enhanced model (48.8%)")
                print("   Progress toward target")
                status = "IMPROVED"
            else:
                print("📊 TRAINING RESULTS NEED ANALYSIS")
                status = "NEEDS_ANALYSIS"
            
            if class_balance_fixed:
                print("✅ Class imbalance successfully resolved")
            
            print()
            
            # Comparison table
            print("📊 MODEL COMPARISON:")
            print(f"   Baseline:   52.0% mIoU (5.9MB)")
            print(f"   Enhanced:   48.8% mIoU (38.2MB) - overfitting issue")
            print(f"   Balanced:   {overall_miou:.1%} mIoU ({size_mb:.1f}MB) - DiceFocal fix")
            
            improvement_vs_baseline = (overall_miou - 0.52) * 100
            improvement_vs_enhanced = (overall_miou - 0.488) * 100
            
            print(f"   vs Baseline: {improvement_vs_baseline:+.1f}% points")
            print(f"   vs Enhanced: {improvement_vs_enhanced:+.1f}% points")
            
            return status, overall_miou
            
        except Exception as e:
            print(f"Results file error: {e}")
            return "ERROR", 0
    else:
        print("📋 Training results not available yet")
        print("   Check if training is in progress or completed without results")
        return "IN_PROGRESS", 0
    
    print()

def check_gpu_status():
    """Check GPU usage for training status."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            
            print("GPU STATUS:")
            print(f"   Allocated: {allocated:.1f}GB")
            print(f"   Reserved: {cached:.1f}GB")
            
            if allocated > 2.0:
                print("   Status: ⚡ TRAINING ACTIVE")
            elif allocated > 0.5:
                print("   Status: 🔄 MODEL LOADED")
            else:
                print("   Status: 💤 IDLE")
        else:
            print("GPU: ❌ CUDA not available")
    except Exception as e:
        print(f"GPU check error: {e}")
    
    print()

def main():
    """Main monitoring function."""
    status, miou = monitor_balanced_training()
    check_gpu_status()
    
    print("🔄 NEXT STEPS:")
    
    if status == "EXCELLENT" or status == "SUCCESS":
        print("1. ✅ Training target achieved!")
        print("2. 📊 Run final evaluation script")
        print("3. 🚀 Deploy to production API")
        print("4. 🧪 Run integration tests")
    elif status == "GOOD" or status == "IMPROVED":
        print("1. ✅ Significant improvement achieved!")
        print("2. 📊 Run comprehensive evaluation")
        print("3. 🤔 Consider: Deploy current model or continue training")
        print("4. 🔧 Optional: Architecture fine-tuning")
    elif status == "IN_PROGRESS":
        print("1. ⏳ Wait for training completion")
        print("2. 🔍 Monitor for another 30-60 minutes")
        print("3. 📊 Check results when available")
    else:
        print("1. 🔍 Investigate training results")
        print("2. 📊 Run detailed evaluation")
        print("3. 🔧 Consider hyperparameter adjustments")
    
    print()
    print("Commands:")
    print("   python scripts/balanced_train.py     # Start/restart training")
    print("   python scripts/balanced_monitor.py   # Check progress") 
    print("   python scripts/balanced_eval.py      # Final evaluation")

if __name__ == "__main__":
    main()