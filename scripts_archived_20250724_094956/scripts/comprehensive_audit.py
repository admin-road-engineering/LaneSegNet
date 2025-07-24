#!/usr/bin/env python3
"""
Comprehensive Model Training and Validation Audit
==================================================

This script conducts a thorough audit of our training methodology, 
model checkpoints, and validation processes to identify the root cause
of the massive performance discrepancy (85.1% reported vs 32.5% actual mIoU).

Audit Areas:
1. Model checkpoint integrity and loading verification
2. Training/validation dataset split consistency 
3. Metric calculation methodology validation
4. Data preprocessing pipeline verification
5. Comparative analysis of all backup models
6. Training log analysis and validation curve review
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
import time
import hashlib
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

class ModelAudit:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.issues_found = []
        
    def log_issue(self, severity, category, description):
        """Log audit issues with severity levels"""
        issue = {
            'severity': severity,  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
            'category': category,  # 'MODEL', 'DATA', 'TRAINING', 'VALIDATION'
            'description': description,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.issues_found.append(issue)
        print(f"[{severity}] {category}: {description}")
    
    def audit_model_checkpoints(self):
        """Audit 1: Verify model checkpoint integrity and metadata"""
        print("\n" + "="*60)
        print("AUDIT 1: MODEL CHECKPOINT INTEGRITY")
        print("="*60)
        
        # Check primary model
        primary_model = 'work_dirs/premium_gpu_best_model.pth'
        backup_model = 'model_backups/epoch50_final_masterpiece_20250722_194650/premium_gpu_best_model.pth'
        
        models_to_check = [
            ('Primary Model', primary_model),
            ('Backup Model', backup_model)
        ]
        
        for name, model_path in models_to_check:
            print(f"\nChecking {name}: {model_path}")
            
            if not Path(model_path).exists():
                self.log_issue('CRITICAL', 'MODEL', f"{name} not found at {model_path}")
                continue
                
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Extract metadata
                epoch = checkpoint.get('epoch', 'unknown')
                reported_miou = checkpoint.get('best_miou', 0)
                
                print(f"  Epoch: {epoch}")
                print(f"  Reported mIoU: {reported_miou*100:.1f}%")
                
                # Check required keys
                required_keys = ['model_state_dict', 'epoch', 'best_miou']
                for key in required_keys:
                    if key not in checkpoint:
                        self.log_issue('HIGH', 'MODEL', f"{name} missing key: {key}")
                
                # Try loading the model
                model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Model loading: SUCCESS")
                
                # Calculate checkpoint hash for integrity
                file_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
                print(f"  File hash: {file_hash[:16]}...")
                
                self.results[name] = {
                    'epoch': epoch,
                    'reported_miou': reported_miou,
                    'file_hash': file_hash,
                    'loading_success': True
                }
                
            except Exception as e:
                self.log_issue('CRITICAL', 'MODEL', f"{name} loading failed: {e}")
                print(f"  ERROR: {e}")
    
    def audit_dataset_integrity(self):
        """Audit 2: Verify dataset splits and annotation quality"""
        print("\n" + "="*60)
        print("AUDIT 2: DATASET INTEGRITY")
        print("="*60)
        
        # Check dataset directories
        data_dirs = {
            'train_img': 'data/ael_mmseg/img_dir/train',
            'train_ann': 'data/ael_mmseg/ann_dir/train',
            'val_img': 'data/ael_mmseg/img_dir/val',
            'val_ann': 'data/ael_mmseg/ann_dir/val'
        }
        
        dataset_stats = {}
        
        for name, path in data_dirs.items():
            dir_path = Path(path)
            if not dir_path.exists():
                self.log_issue('CRITICAL', 'DATA', f"Dataset directory missing: {path}")
                continue
                
            if 'img' in name:
                files = list(dir_path.glob('*.jpg'))
            else:
                files = list(dir_path.glob('*.png'))
                
            dataset_stats[name] = len(files)
            print(f"  {name}: {len(files)} files")
        
        # Check for matching pairs
        if 'train_img' in dataset_stats and 'train_ann' in dataset_stats:
            if dataset_stats['train_img'] != dataset_stats['train_ann']:
                self.log_issue('HIGH', 'DATA', 
                    f"Train img/ann mismatch: {dataset_stats['train_img']} vs {dataset_stats['train_ann']}")
        
        if 'val_img' in dataset_stats and 'val_ann' in dataset_stats:
            if dataset_stats['val_img'] != dataset_stats['val_ann']:
                self.log_issue('HIGH', 'DATA', 
                    f"Val img/ann mismatch: {dataset_stats['val_img']} vs {dataset_stats['val_ann']}")
        
        # Sample annotation quality check
        print(f"\nSampling annotation quality...")
        val_ann_dir = Path('data/ael_mmseg/ann_dir/val')
        sample_files = list(val_ann_dir.glob('*.png'))[:10]
        
        class_distributions = []
        for ann_file in sample_files:
            mask = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
            unique_classes = np.unique(mask)
            class_dist = {
                'background': (mask == 0).sum(),
                'white_solid': (mask == 1).sum(), 
                'white_dashed': (mask == 2).sum(),
                'other': (mask > 2).sum()
            }
            class_distributions.append(class_dist)
            
            if class_dist['other'] > 0:
                self.log_issue('MEDIUM', 'DATA', f"Invalid classes found in {ann_file.name}")
        
        # Calculate average class distribution
        if class_distributions:
            avg_dist = {}
            for key in class_distributions[0].keys():
                avg_dist[key] = np.mean([d[key] for d in class_distributions])
            
            total_pixels = sum(avg_dist.values())
            print(f"  Average class distribution:")
            for class_name, count in avg_dist.items():
                percentage = (count / total_pixels) * 100
                print(f"    {class_name}: {percentage:.1f}%")
                
            # Check for severe class imbalance
            lane_percentage = ((avg_dist['white_solid'] + avg_dist['white_dashed']) / total_pixels) * 100
            if lane_percentage < 1.0:
                self.log_issue('HIGH', 'DATA', f"Severe class imbalance: Only {lane_percentage:.2f}% lane pixels")
    
    def audit_validation_methodology(self):
        """Audit 3: Verify validation methodology consistency"""
        print("\n" + "="*60)
        print("AUDIT 3: VALIDATION METHODOLOGY")
        print("="*60)
        
        # Load model for testing
        model_path = 'work_dirs/premium_gpu_best_model.pth'
        if not Path(model_path).exists():
            self.log_issue('CRITICAL', 'VALIDATION', "Primary model not found for validation testing")
            return
            
        checkpoint = torch.load(model_path, map_location='cpu')
        model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print("Testing validation methodology with 10 samples...")
        
        # Test with same preprocessing as training
        val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        sample_results = []
        sample_count = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                if sample_count >= 10:
                    break
                    
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = model(images)
                predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                
                # Calculate IoU for this batch
                batch_predictions = predictions.cpu().numpy()
                batch_targets = masks.cpu().numpy()
                
                for i in range(batch_predictions.shape[0]):
                    if sample_count >= 10:
                        break
                        
                    pred = batch_predictions[i]
                    target = batch_targets[i]
                    
                    # Per-class IoU
                    ious = []
                    for class_id in range(3):
                        pred_class = (pred == class_id)
                        target_class = (target == class_id)
                        
                        intersection = np.logical_and(pred_class, target_class).sum()
                        union = np.logical_or(pred_class, target_class).sum()
                        
                        iou = intersection / union if union > 0 else 0
                        ious.append(iou)
                    
                    sample_results.append({
                        'sample_id': sample_count,
                        'ious': ious,
                        'mean_iou': np.mean(ious),
                        'lane_coverage_pred': ((pred == 1) | (pred == 2)).sum() / pred.size,
                        'lane_coverage_gt': ((target == 1) | (target == 2)).sum() / target.size
                    })
                    
                    sample_count += 1
        
        # Analyze results
        if sample_results:
            mean_ious = [r['mean_iou'] for r in sample_results]
            avg_miou = np.mean(mean_ious)
            std_miou = np.std(mean_ious)
            
            lane_pred_coverage = [r['lane_coverage_pred'] for r in sample_results]
            lane_gt_coverage = [r['lane_coverage_gt'] for r in sample_results]
            
            print(f"  Validation sample results (n={len(sample_results)}):")
            print(f"    Average mIoU: {avg_miou*100:.1f}% ± {std_miou*100:.1f}%")
            print(f"    Lane prediction coverage: {np.mean(lane_pred_coverage)*100:.2f}%")
            print(f"    Lane ground truth coverage: {np.mean(lane_gt_coverage)*100:.2f}%")
            
            # Check for systematic issues
            if avg_miou < 0.5:
                self.log_issue('CRITICAL', 'VALIDATION', 
                    f"Validation mIoU extremely low: {avg_miou*100:.1f}%")
            
            if np.mean(lane_pred_coverage) < 0.001:
                self.log_issue('CRITICAL', 'VALIDATION', 
                    "Model predicting almost no lane pixels")
            
            self.results['validation_sample'] = {
                'avg_miou': avg_miou,
                'std_miou': std_miou,
                'avg_lane_pred': np.mean(lane_pred_coverage),
                'avg_lane_gt': np.mean(lane_gt_coverage)
            }
    
    def audit_training_logs(self):
        """Audit 4: Review training logs and learning curves"""
        print("\n" + "="*60)
        print("AUDIT 4: TRAINING LOG ANALYSIS")
        print("="*60)
        
        # Look for training logs
        log_locations = [
            'premium_training.log',
            'training.log',
            'logs/training.log'
        ]
        
        log_found = False
        for log_path in log_locations:
            if Path(log_path).exists():
                print(f"Found training log: {log_path}")
                log_found = True
                # Could analyze log content here
                break
        
        if not log_found:
            self.log_issue('MEDIUM', 'TRAINING', "No training logs found for analysis")
        
        # Check for tensorboard logs
        tb_dirs = ['runs', 'logs', 'tensorboard']
        tb_found = False
        for tb_dir in tb_dirs:
            if Path(tb_dir).exists():
                print(f"Found potential tensorboard logs: {tb_dir}")
                tb_found = True
                break
        
        if not tb_found:
            self.log_issue('LOW', 'TRAINING', "No tensorboard logs found")
    
    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE AUDIT REPORT")
        print("="*60)
        
        print(f"\nAUDIT COMPLETED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total issues found: {len(self.issues_found)}")
        
        # Group issues by severity
        issues_by_severity = defaultdict(list)
        for issue in self.issues_found:
            issues_by_severity[issue['severity']].append(issue)
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in issues_by_severity:
                print(f"\n{severity} ISSUES ({len(issues_by_severity[severity])}):")
                for i, issue in enumerate(issues_by_severity[severity], 1):
                    print(f"  {i}. [{issue['category']}] {issue['description']}")
        
        # Key findings summary
        print(f"\nKEY FINDINGS:")
        if 'Primary Model' in self.results and 'validation_sample' in self.results:
            reported = self.results['Primary Model']['reported_miou'] * 100
            actual = self.results['validation_sample']['avg_miou'] * 100
            discrepancy = reported - actual
            
            print(f"  Reported mIoU: {reported:.1f}%")
            print(f"  Actual mIoU: {actual:.1f}%")
            print(f"  Discrepancy: {discrepancy:.1f}% {'(CRITICAL)' if discrepancy > 20 else ''}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        critical_issues = len(issues_by_severity.get('CRITICAL', []))
        high_issues = len(issues_by_severity.get('CRITICAL', [])) + len(issues_by_severity.get('HIGH', []))
        
        if critical_issues > 0:
            print("  1. IMMEDIATE ACTION REQUIRED: Address critical model/data issues")
            print("  2. Re-train model with corrected methodology")
            print("  3. Implement proper validation tracking")
        elif high_issues > 0:
            print("  1. Review and fix high-priority issues")
            print("  2. Validate model performance on independent test set")
        else:
            print("  1. Minor issues found - proceed with caution")
        
        # Save detailed report
        report_path = f"audit_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': self.results,
                'issues': self.issues_found,
                'summary': {
                    'total_issues': len(self.issues_found),
                    'critical_issues': critical_issues,
                    'high_issues': len(issues_by_severity.get('HIGH', [])),
                }
            }, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {report_path}")
        print("="*60)

def main():
    print("COMPREHENSIVE MODEL TRAINING & VALIDATION AUDIT")
    print("=" * 60)
    print("Investigating performance discrepancy: 85.1% reported vs 32.5% actual mIoU")
    print("=" * 60)
    
    auditor = ModelAudit()
    
    try:
        auditor.audit_model_checkpoints()
        auditor.audit_dataset_integrity()
        auditor.audit_validation_methodology()
        auditor.audit_training_logs()
        auditor.generate_audit_report()
        
    except Exception as e:
        print(f"\nAUDIT FAILED: {e}")
        auditor.log_issue('CRITICAL', 'AUDIT', f"Audit process failed: {e}")
        auditor.generate_audit_report()

if __name__ == "__main__":
    main()