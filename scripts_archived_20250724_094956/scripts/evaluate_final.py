#!/usr/bin/env python3
"""
Final evaluation of Phase 3.2 enhanced training results
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime

def check_final_results():
    """Check if we have final training results."""
    print("=== Phase 3.2: Final Results Check ===")
    print(f"Evaluation time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    work_dir = Path("work_dirs")
    
    # Check enhanced model
    enhanced_model = work_dir / "enhanced_best_model.pth"
    if not enhanced_model.exists():
        print("❌ Enhanced model not found. Training may not be complete.")
        return False
    
    size_mb = enhanced_model.stat().st_size / 1024**2
    mod_time = datetime.fromtimestamp(enhanced_model.stat().st_mtime)
    
    print(f"✅ Enhanced Model Found:")
    print(f"   Size: {size_mb:.1f}MB")
    print(f"   Created: {mod_time.strftime('%H:%M:%S')}")
    print()
    
    # Try to load model to verify it's valid
    try:
        # Define the enhanced model architecture
        class EnhancedLaneNet(nn.Module):
            def __init__(self, num_classes=4):
                super().__init__()
                self.num_classes = num_classes
                
                # Enhanced encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                # Enhanced decoder
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, num_classes, 1)
                )
            
            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return output
        
        model = EnhancedLaneNet(num_classes=4)
        state_dict = torch.load(enhanced_model, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✅ Model Loading Successful:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print()
        
    except Exception as e:
        print(f"⚠ Model loading issue: {e}")
        print()
    
    # Check for training results
    results_file = work_dir / "enhanced_training_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print("📊 TRAINING RESULTS:")
            miou = results.get('best_miou', 0)
            print(f"   Best mIoU: {miou:.1%}")
            print(f"   Epochs Completed: {results.get('num_epochs_completed', 'Unknown')}")
            print(f"   Training Time: {results.get('training_time_hours', 0):.1f} hours")
            print(f"   Architecture: {results.get('architecture', 'Enhanced CNN')}")
            print()
            
            # Check target achievement
            target_80 = results.get('target_80_achieved', False)
            target_85 = results.get('target_85_achieved', False)
            
            if target_85:
                print("🏆 OUTSTANDING! 85%+ TARGET ACHIEVED!")
                print("   Ready for production deployment")
                status = "EXCELLENT"
            elif target_80:
                print("🎯 SUCCESS! 80-85% TARGET ACHIEVED!")
                print("   Ready for production deployment") 
                status = "SUCCESS"
            elif miou >= 0.65:
                print("✅ GOOD PROGRESS! Above baseline target")
                print("   Significant improvement over baseline (52%)")
                status = "IMPROVED"
            else:
                print("⚠ Below target, but may have improved over baseline")
                status = "PARTIAL"
            
            return status, miou
            
        except Exception as e:
            print(f"Results file error: {e}")
            return "UNKNOWN", 0
    else:
        print("📋 Training results file not found")
        print("   Training may have completed without saving final results")
        print("   Model appears trained based on file size and timestamp")
        return "COMPLETED_NO_RESULTS", 0

def estimate_performance():
    """Estimate performance based on model characteristics."""
    print()
    print("📈 PERFORMANCE ESTIMATE:")
    print("   Baseline (Simple CNN): 52% mIoU")
    print("   Enhanced (Deep CNN + BatchNorm + Focal Loss): ?")
    print()
    print("Expected improvements from enhancements:")
    print("   + BatchNormalization: +5-10% mIoU")
    print("   + Deeper architecture: +10-15% mIoU") 
    print("   + Focal Loss: +5-8% mIoU")
    print("   + Enhanced augmentations: +3-5% mIoU")
    print()
    print("   Estimated range: 75-85% mIoU")

def main():
    """Main evaluation function."""
    status, miou = check_final_results()
    
    if status == "COMPLETED_NO_RESULTS":
        estimate_performance()
    
    print()
    print("🔄 NEXT STEPS:")
    
    if status in ["EXCELLENT", "SUCCESS"]:
        print("1. ✅ Training target achieved!")
        print("2. 🚀 Deploy model to production API")
        print("3. 🧪 Run integration tests")
        print("4. 📊 Validate inference performance (<1000ms)")
    elif status == "IMPROVED":
        print("1. ✅ Significant improvement over baseline")
        print("2. 🤔 Consider additional training or architecture tweaks")
        print("3. 🚀 Could proceed with current model if acceptable")
    else:
        print("1. 🔍 Investigate training results")
        print("2. 🧪 Run full evaluation on validation set")
        print("3. 🔄 Consider additional training if needed")
    
    print()
    print("To proceed with integration:")
    print("   python scripts/integrate_enhanced_model.py")

if __name__ == "__main__":
    main()