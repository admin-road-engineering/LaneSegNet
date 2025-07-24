#!/usr/bin/env python3
"""
Comprehensive Model Validation - Test ALL backup models against consistent validation set
Identify the true best performing model with proper metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
from datetime import datetime
import time

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

class ComprehensiveModelValidator:
    """Comprehensive validation of all model checkpoints"""
    
    def __init__(self, validation_samples: int = 500):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_samples = validation_samples
        
        # Load validation dataset (consistent across all models)
        self.val_dataset = PremiumDataset(
            'data/ael_mmseg/img_dir/val', 
            'data/ael_mmseg/ann_dir/val', 
            mode='val'
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Validator initialized with {len(self.val_dataset)} validation samples")
        print(f"Testing on {min(validation_samples, len(self.val_dataset))} samples per model")
    
    def load_model(self, checkpoint_path: Path) -> tuple:
        """Load model from checkpoint and return model + metadata"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model
            model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # Extract metadata
            metadata = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'reported_miou': checkpoint.get('best_miou', 0),
                'reported_balanced_score': checkpoint.get('best_balanced_score', 0),
                'approach': checkpoint.get('approach', 'unknown'),
                'checkpoint_path': str(checkpoint_path)
            }
            
            return model, metadata
            
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")
            return None, None
    
    def evaluate_model(self, model, metadata: dict) -> dict:
        """Evaluate a single model and return comprehensive metrics"""
        print(f"  Evaluating {Path(metadata['checkpoint_path']).parent.name}...")
        
        all_predictions = []
        all_targets = []
        sample_count = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                if sample_count >= self.validation_samples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = model(images)
                predictions = torch.softmax(outputs, dim=1).argmax(dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                sample_count += images.size(0)
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / sample_count * 1000  # ms per image
        
        # Calculate comprehensive metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Per-class IoU
        class_names = ['background', 'white_solid', 'white_dashed']
        ious = {}
        
        for class_id, class_name in enumerate(class_names):
            pred_class = (all_predictions == class_id)
            target_class = (all_targets == class_id)
            
            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious[class_name] = iou
        
        # Calculate additional metrics
        overall_miou = np.mean(list(ious.values()))
        lane_miou = np.mean([ious['white_solid'], ious['white_dashed']])
        
        # Pixel accuracy
        correct_pixels = (all_predictions == all_targets).sum()
        total_pixels = all_targets.size
        pixel_accuracy = correct_pixels / total_pixels
        
        # Lane detection coverage
        pred_lane_pixels = ((all_predictions == 1) | (all_predictions == 2)).sum()
        target_lane_pixels = ((all_targets == 1) | (all_targets == 2)).sum()
        lane_recall = pred_lane_pixels / target_lane_pixels if target_lane_pixels > 0 else 0
        
        # Precision for lanes
        if pred_lane_pixels > 0:
            correct_lane_pixels = np.logical_and(
                (all_predictions == 1) | (all_predictions == 2),
                (all_targets == 1) | (all_targets == 2)
            ).sum()
            lane_precision = correct_lane_pixels / pred_lane_pixels
        else:
            lane_precision = 0
        
        # F1 score for lanes
        if lane_precision + lane_recall > 0:
            lane_f1 = 2 * (lane_precision * lane_recall) / (lane_precision + lane_recall)
        else:
            lane_f1 = 0
        
        results = {
            'model_info': metadata,
            'samples_tested': sample_count,
            'inference_time_ms': avg_inference_time,
            
            # IoU metrics
            'background_iou': ious['background'],
            'white_solid_iou': ious['white_solid'],
            'white_dashed_iou': ious['white_dashed'],
            'overall_miou': overall_miou,
            'lane_miou': lane_miou,
            
            # Additional metrics
            'pixel_accuracy': pixel_accuracy,
            'lane_recall': lane_recall,
            'lane_precision': lane_precision,
            'lane_f1': lane_f1,
            
            # Comparison with reported performance
            'reported_vs_actual_diff': metadata['reported_miou'] - overall_miou if metadata['reported_miou'] > 0 else 0
        }
        
        return results
    
    def find_all_model_checkpoints(self) -> list:
        """Find all model checkpoints in backup directories"""
        backup_dir = Path('model_backups')
        checkpoints = []
        
        if not backup_dir.exists():
            print("No model_backups directory found")
            return checkpoints
        
        # Find all .pth files in backup directories
        for backup_folder in backup_dir.iterdir():
            if backup_folder.is_dir():
                pth_files = list(backup_folder.glob('*.pth'))
                for pth_file in pth_files:
                    checkpoints.append(pth_file)
        
        # Also check work_dirs
        work_dir = Path('work_dirs')
        if work_dir.exists():
            work_checkpoints = list(work_dir.glob('*.pth'))
            checkpoints.extend(work_checkpoints)
        
        print(f"Found {len(checkpoints)} model checkpoints")
        return sorted(checkpoints)
    
    def run_comprehensive_validation(self) -> dict:
        """Run validation on all models and return ranked results"""
        print("=" * 80)
        print("COMPREHENSIVE MODEL VALIDATION")
        print("Testing ALL backup models against consistent validation set")
        print("=" * 80)
        
        checkpoints = self.find_all_model_checkpoints()
        
        if not checkpoints:
            print("No model checkpoints found!")
            return {}
        
        results = []
        
        for i, checkpoint_path in enumerate(checkpoints):
            print(f"\\n[{i+1}/{len(checkpoints)}] Testing: {checkpoint_path.name}")
            
            model, metadata = self.load_model(checkpoint_path)
            if model is None:
                continue
            
            try:
                result = self.evaluate_model(model, metadata)
                results.append(result)
                
                # Quick summary
                print(f"    Actual mIoU: {result['overall_miou']*100:.1f}% (reported: {metadata['reported_miou']*100:.1f}%)")
                print(f"    Lane mIoU: {result['lane_miou']*100:.1f}%")
                print(f"    Inference: {result['inference_time_ms']:.1f}ms/image")
                
            except Exception as e:
                print(f"    ERROR during evaluation: {e}")
                continue
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
        
        return self.analyze_results(results)
    
    def analyze_results(self, results: list) -> dict:
        """Analyze and rank all model results"""
        if not results:
            print("No valid results to analyze!")
            return {}
        
        print("\\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION RESULTS")
        print("=" * 80)
        
        # Sort by overall mIoU (descending)
        results_sorted = sorted(results, key=lambda x: x['overall_miou'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<35} {'Actual mIoU':<12} {'Lane mIoU':<10} {'Reported':<10} {'Diff':<8} {'Speed':<8}")
        print("-" * 95)
        
        for i, result in enumerate(results_sorted):
            model_name = Path(result['model_info']['checkpoint_path']).parent.name
            if len(model_name) > 33:
                model_name = model_name[:30] + "..."
            
            actual_miou = result['overall_miou'] * 100
            lane_miou = result['lane_miou'] * 100
            reported_miou = result['model_info']['reported_miou'] * 100
            diff = result['reported_vs_actual_diff'] * 100
            speed = result['inference_time_ms']
            
            print(f"{i+1:<4} {model_name:<35} {actual_miou:<12.1f} {lane_miou:<10.1f} {reported_miou:<10.1f} {diff:<8.1f} {speed:<8.1f}")
        
        # Best model analysis
        best_model = results_sorted[0]
        print(f"\\nBEST PERFORMING MODEL:")
        print(f"  Path: {best_model['model_info']['checkpoint_path']}")
        print(f"  Actual mIoU: {best_model['overall_miou']*100:.1f}%")
        print(f"  Lane mIoU: {best_model['lane_miou']*100:.1f}%")
        print(f"  Background IoU: {best_model['background_iou']*100:.1f}%")
        print(f"  White Solid IoU: {best_model['white_solid_iou']*100:.1f}%")
        print(f"  White Dashed IoU: {best_model['white_dashed_iou']*100:.1f}%")
        print(f"  Lane F1 Score: {best_model['lane_f1']*100:.1f}%")
        print(f"  Inference Speed: {best_model['inference_time_ms']:.1f}ms/image")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'validation_summary': {
                    'total_models_tested': len(results),
                    'validation_samples': self.validation_samples,
                    'best_model': best_model['model_info']['checkpoint_path'],
                    'best_actual_miou': best_model['overall_miou'],
                    'timestamp': timestamp
                },
                'detailed_results': results_sorted
            }, f, indent=2)
        
        print(f"\\nDetailed results saved to: {results_file}")
        print("=" * 80)
        
        return {
            'best_model': best_model,
            'all_results': results_sorted,
            'summary_file': results_file
        }

def main():
    validator = ComprehensiveModelValidator(validation_samples=500)
    results = validator.run_comprehensive_validation()
    
    if results and 'best_model' in results:
        best_model_path = results['best_model']['model_info']['checkpoint_path']
        print(f"\\nRECOMMENDATION: Use {Path(best_model_path).name} as the production model")

if __name__ == "__main__":
    main()