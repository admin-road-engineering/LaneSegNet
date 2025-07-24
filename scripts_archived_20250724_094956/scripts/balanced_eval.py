#!/usr/bin/env python3
"""
Comprehensive evaluation of the balanced model with DiceFocal loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Import the optimized model architecture
class OptimizedLaneNet(nn.Module):
    """Same architecture as in balanced_train.py"""
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.7),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.7),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.3),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class EvalDataset:
    """Evaluation dataset."""
    def __init__(self, img_dir, mask_dir, num_samples=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
        
        if num_samples:
            self.images = self.images[:num_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        mask = np.clip(mask, 0, 3)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask, img_path.stem

def calculate_detailed_iou(pred, target, num_classes=4):
    """Calculate detailed IoU metrics."""
    ious = []
    precisions = []
    recalls = []
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # IoU calculation
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        # Precision and recall
        if pred_cls.sum() > 0:
            precision = intersection / pred_cls.sum()
        else:
            precision = 0.0
        
        if target_cls.sum() > 0:
            recall = intersection / target_cls.sum()
        else:
            recall = 1.0 if pred_cls.sum() == 0 else 0.0
        
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
    
    return ious, precisions, recalls

def evaluate_balanced_model(dataset_split='val', num_samples=200):
    """Comprehensive evaluation of balanced model."""
    print("=== Phase 3.2.5: Balanced Model Evaluation ===")
    print(f"Dataset: {dataset_split}")
    print(f"Max samples: {num_samples}")
    print(f"Evaluation time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load balanced model
    model_path = "work_dirs/balanced_best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Balanced model not found: {model_path}")
        print("Run training first: python scripts/balanced_train.py")
        return None
    
    # Initialize model
    model = OptimizedLaneNet(num_classes=4, dropout_rate=0.3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Balanced model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None
    
    # Model info
    size_mb = Path(model_path).stat().st_size / 1024**2
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {size_mb:.1f}MB")
    print(f"Parameters: {total_params:,}")
    print()
    
    # Load dataset
    img_dir = f"data/ael_mmseg/img_dir/{dataset_split}"
    mask_dir = f"data/ael_mmseg/ann_dir/{dataset_split}"
    
    dataset = EvalDataset(img_dir, mask_dir, num_samples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print(f"Evaluating {len(dataset)} samples...")
    print()
    
    # Evaluation
    all_ious = []
    all_precisions = []
    all_recalls = []
    class_names = ['background', 'white_solid', 'white_dashed', 'yellow_solid']
    
    inference_times = []
    
    with torch.no_grad():
        for images, masks, names in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = model(images)
            end_time.record()
            
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # milliseconds
            inference_times.extend([inference_time / images.size(0)] * images.size(0))
            
            pred = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                ious, precisions, recalls = calculate_detailed_iou(pred[i], masks[i])
                all_ious.append(ious)
                all_precisions.append(precisions)
                all_recalls.append(recalls)
    
    # Calculate metrics
    mean_ious = np.mean(all_ious, axis=0)
    mean_precisions = np.mean(all_precisions, axis=0)
    mean_recalls = np.mean(all_recalls, axis=0)
    
    overall_miou = np.mean(mean_ious)
    lane_classes_miou = np.mean(mean_ious[1:])  # Exclude background
    
    avg_inference_time = np.mean(inference_times)
    
    # Results
    print()
    print("=== BALANCED MODEL RESULTS ===")
    print(f"Samples evaluated: {len(all_ious)}")
    print(f"Overall mIoU: {overall_miou:.1%}")
    print(f"Lane Classes mIoU: {lane_classes_miou:.1%}")
    print(f"Average inference time: {avg_inference_time:.1f}ms")
    print()
    
    print("📊 DETAILED PER-CLASS METRICS:")
    print(f"{'Class':<15} {'IoU':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 45)
    
    for i, name in enumerate(class_names):
        iou = mean_ious[i]
        precision = mean_precisions[i]
        recall = mean_recalls[i]
        
        status = "✅" if iou > 0.5 else ("⚠" if iou > 0.2 else "❌")
        print(f"{status} {name:<13} {iou:.1%}    {precision:.1%}      {recall:.1%}")
    
    print()
    
    # Class imbalance assessment
    min_lane_iou = min(mean_ious[1:]) if len(mean_ious) > 1 else 0
    
    if min_lane_iou > 0.5:
        print("🎯 CLASS IMBALANCE FIXED: All lane classes >50% IoU")
        imbalance_status = "FIXED"
    elif min_lane_iou > 0.3:
        print("📈 CLASS IMBALANCE IMPROVING: Min lane IoU {:.1%}".format(min_lane_iou))
        imbalance_status = "IMPROVING"
    else:
        print("⚠ CLASS IMBALANCE PERSISTS: Min lane IoU {:.1%}".format(min_lane_iou))
        imbalance_status = "PERSISTS"
    
    print()
    
    # Performance targets
    print("🎯 TARGET ASSESSMENT:")
    if overall_miou >= 0.85:
        print("🏆 OUTSTANDING! 85%+ target achieved!")
        target_status = "EXCELLENT"
    elif overall_miou >= 0.80:
        print("🎯 SUCCESS! 80-85% target achieved!")
        target_status = "SUCCESS"
    elif overall_miou >= 0.70:
        print("✅ GOOD! 70%+ target achieved!")
        target_status = "GOOD"
    elif overall_miou >= 0.60:
        print("📈 IMPROVED! Above baseline (52%)")
        target_status = "IMPROVED"
    else:
        print("📊 NEEDS ANALYSIS: Below expectations")
        target_status = "NEEDS_ANALYSIS"
    
    # Inference time assessment
    if avg_inference_time < 1000:
        print(f"⚡ INFERENCE TIME: {avg_inference_time:.1f}ms (✅ <1000ms target)")
    else:
        print(f"⏱ INFERENCE TIME: {avg_inference_time:.1f}ms (⚠ >1000ms target)")
    
    print()
    
    # Comparison with previous models
    print("📈 MODEL COMPARISON:")
    print(f"   Baseline (Simple):    52.0% mIoU")
    print(f"   Enhanced (Deep):      48.8% mIoU (class imbalance)")
    print(f"   Balanced (DiceFocal): {overall_miou:.1%} mIoU")
    
    improvement_vs_baseline = (overall_miou - 0.52) * 100
    improvement_vs_enhanced = (overall_miou - 0.488) * 100
    
    print(f"   vs Baseline: {improvement_vs_baseline:+.1f}% points")
    print(f"   vs Enhanced: {improvement_vs_enhanced:+.1f}% points")
    print()
    
    # Save detailed results
    detailed_results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset_split': dataset_split,
        'samples_evaluated': len(all_ious),
        'model_size_mb': size_mb,
        'model_parameters': total_params,
        'overall_miou': overall_miou,
        'lane_classes_miou': lane_classes_miou,
        'per_class_ious': mean_ious.tolist(),
        'per_class_precisions': mean_precisions.tolist(),
        'per_class_recalls': mean_recalls.tolist(),
        'class_names': class_names,
        'avg_inference_time_ms': avg_inference_time,
        'target_status': target_status,
        'imbalance_status': imbalance_status,
        'target_80_achieved': overall_miou >= 0.80,
        'target_85_achieved': overall_miou >= 0.85,
        'inference_under_1000ms': avg_inference_time < 1000,
        'class_imbalance_fixed': min_lane_iou > 0.5,
        'improvement_vs_baseline': improvement_vs_baseline,
        'improvement_vs_enhanced': improvement_vs_enhanced
    }
    
    results_path = f"work_dirs/balanced_evaluation_{dataset_split}.json"
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"📊 Detailed results saved: {results_path}")
    
    return detailed_results

def main():
    """Main evaluation function."""
    
    # Run validation evaluation
    print("Running validation set evaluation...")
    val_results = evaluate_balanced_model('val', num_samples=200)
    
    if val_results is None:
        return
    
    print("\n" + "="*60)
    
    # Summary
    print("\n🎯 PHASE 3.2.5 SUMMARY:")
    print(f"✅ DiceFocal Loss Implementation: Complete")
    print(f"📊 Balanced Model Performance: {val_results['overall_miou']:.1%} mIoU")
    print(f"🎯 Target Achievement: {val_results['target_status']}")
    print(f"⚖ Class Imbalance Status: {val_results['imbalance_status']}")
    print(f"⚡ Inference Performance: {val_results['avg_inference_time_ms']:.1f}ms")
    
    print(f"\n🔄 NEXT STEPS:")
    
    if val_results['target_status'] in ['EXCELLENT', 'SUCCESS']:
        print("1. ✅ Training target achieved!")
        print("2. 📊 Run test set evaluation")
        print("3. 🚀 Deploy to production API")
        print("4. 🧪 Run integration tests")
    elif val_results['target_status'] in ['GOOD', 'IMPROVED']:
        print("1. ✅ Significant improvement!")
        print("2. 📊 Consider production deployment")
        print("3. 🔧 Optional: Further hyperparameter tuning")
    else:
        print("1. 🔍 Analyze training parameters")
        print("2. 🔧 Consider architecture adjustments")
        print("3. 📊 Additional training iterations")
    
    # Offer test evaluation if validation looks good
    if val_results['overall_miou'] >= 0.70:
        print("\n❓ Run test set evaluation? (final validation)")
        print("   python scripts/balanced_eval.py --test")

if __name__ == "__main__":
    import sys
    
    if '--test' in sys.argv:
        print("Running test set evaluation...")
        test_results = evaluate_balanced_model('test', num_samples=100)
    else:
        main()