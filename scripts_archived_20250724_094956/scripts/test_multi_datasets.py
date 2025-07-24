#!/usr/bin/env python3
"""
Multi-Dataset Testing
====================

Test the optimized 78.2% mIoU model on different datasets:
1. Original AEL validation set (baseline)
2. Australian aerial images
3. SS_Dense dataset  
4. SS_Multi_Lane dataset

This will reveal performance across different geographic regions and data types.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet, PremiumDataset

class MultiDatasetTester:
    def __init__(self, model_path='work_dirs/premium_gpu_best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
        # Image preprocessing (same as training)
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def load_model(self, model_path):
        """Load the optimized model"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        reported_miou = checkpoint.get('best_miou', 0)
        print(f"Model loaded: Reported mIoU {reported_miou*100:.1f}%")
        return model
        
    def calculate_iou(self, pred_mask, gt_mask):
        """Calculate IoU for 3-class system"""
        pred_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
        gt_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
        
        class_names = ['background', 'white_solid', 'white_dashed']
        ious = []
        
        for class_id in range(3):
            pred_class = (pred_np == class_id)
            gt_class = (gt_np == class_id)
            
            intersection = np.logical_and(pred_class, gt_class).sum()
            union = np.logical_or(pred_class, gt_class).sum()
            
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        return ious, class_names
    
    def test_ael_validation(self, max_samples=100):
        """Test on original AEL validation set (baseline)"""
        print(f"\nTesting AEL Validation Set (baseline)")
        print("-" * 40)
        
        val_dataset = PremiumDataset('data/ael_mmseg/img_dir/val', 'data/ael_mmseg/ann_dir/val', mode='val')
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        all_ious = []
        sample_count = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                if sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for i in range(images.size(0)):
                    if sample_count >= max_samples:
                        break
                    
                    ious, class_names = self.calculate_iou(predictions[i], masks[i])
                    all_ious.append(ious)
                    sample_count += 1
        
        return self.analyze_results("AEL Validation", all_ious, class_names, sample_count)
    
    def test_australian_images(self, max_samples=20):
        """Test on Australian aerial images from Downloads"""
        print(f"\nTesting Australian Aerial Images")
        print("-" * 40)
        
        # Find Australian test images
        downloads_dir = Path("C:/Users/Admin/Downloads")
        australian_images = list(downloads_dir.glob("Australia_latest_*.png"))
        
        if not australian_images:
            print("No Australian images found in Downloads")
            return None
            
        print(f"Found {len(australian_images)} Australian images")
        
        # Test on available images (no ground truth, so visual analysis only)
        all_predictions = []
        lane_coverages = []
        
        with torch.no_grad():
            for i, img_path in enumerate(australian_images[:max_samples]):
                print(f"  Processing {img_path.name}...")
                
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing
                transformed = self.transform(image=image)
                input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                
                # Predict
                outputs = self.model(input_tensor)
                prediction = torch.argmax(outputs, dim=1)[0]
                confidence = torch.max(torch.softmax(outputs, dim=1), dim=1)[0].mean()
                
                # Calculate lane coverage
                pred_np = prediction.cpu().numpy()
                lane_pixels = ((pred_np == 1) | (pred_np == 2)).sum()
                total_pixels = pred_np.size
                lane_coverage = lane_pixels / total_pixels
                
                all_predictions.append(prediction.cpu().numpy())
                lane_coverages.append(lane_coverage)
                
                print(f"    Lane coverage: {lane_coverage*100:.2f}%, Confidence: {confidence:.3f}")
        
        # Analysis
        avg_coverage = np.mean(lane_coverages)
        std_coverage = np.std(lane_coverages)
        
        print(f"\nAustralian Images Analysis:")
        print(f"  Images tested: {len(lane_coverages)}")
        print(f"  Average lane coverage: {avg_coverage*100:.2f}% ± {std_coverage*100:.2f}%")
        print(f"  Range: {min(lane_coverages)*100:.2f}% - {max(lane_coverages)*100:.2f}%")
        
        return {
            'dataset': 'Australian Images',
            'samples': len(lane_coverages),
            'avg_lane_coverage': avg_coverage,
            'coverage_std': std_coverage,
            'predictions': all_predictions
        }
    
    def test_ss_dataset(self, dataset_name, max_samples=50):
        """Test on SS_Dense or SS_Multi_Lane dataset"""
        print(f"\nTesting {dataset_name} Dataset")
        print("-" * 40)
        
        # Find dataset directory
        dataset_path = Path(f"data/{dataset_name}")
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return None
        
        # Look for images
        img_patterns = ['*.jpg', '*.png', '*.jpeg']
        image_files = []
        for pattern in img_patterns:
            image_files.extend(list(dataset_path.rglob(pattern)))
        
        if not image_files:
            print(f"No images found in {dataset_path}")
            return None
            
        print(f"Found {len(image_files)} images in {dataset_name}")
        
        # Sample random images for testing
        test_images = random.sample(image_files, min(max_samples, len(image_files)))
        
        lane_coverages = []
        confidence_scores = []
        
        with torch.no_grad():
            for i, img_path in enumerate(test_images):
                try:
                    # Load and preprocess image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply preprocessing
                    transformed = self.transform(image=image)
                    input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                    
                    # Predict
                    outputs = self.model(input_tensor)
                    prediction = torch.argmax(outputs, dim=1)[0]
                    confidence = torch.max(torch.softmax(outputs, dim=1), dim=1)[0].mean()
                    
                    # Calculate lane coverage
                    pred_np = prediction.cpu().numpy()
                    lane_pixels = ((pred_np == 1) | (pred_np == 2)).sum()
                    total_pixels = pred_np.size
                    lane_coverage = lane_pixels / total_pixels
                    
                    lane_coverages.append(lane_coverage)
                    confidence_scores.append(confidence.cpu().item())
                    
                    if i % 10 == 0:
                        print(f"  Processed {i+1}/{len(test_images)} images...")
                        
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
                    continue
        
        if not lane_coverages:
            print(f"No valid predictions for {dataset_name}")
            return None
        
        # Analysis
        avg_coverage = np.mean(lane_coverages)
        std_coverage = np.std(lane_coverages)
        avg_confidence = np.mean(confidence_scores)
        
        print(f"\n{dataset_name} Analysis:")
        print(f"  Images tested: {len(lane_coverages)}")
        print(f"  Average lane coverage: {avg_coverage*100:.2f}% ± {std_coverage*100:.2f}%")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Coverage range: {min(lane_coverages)*100:.2f}% - {max(lane_coverages)*100:.2f}%")
        
        return {
            'dataset': dataset_name,
            'samples': len(lane_coverages),
            'avg_lane_coverage': avg_coverage,
            'coverage_std': std_coverage,
            'avg_confidence': avg_confidence,
            'min_coverage': min(lane_coverages),
            'max_coverage': max(lane_coverages)
        }
    
    def analyze_results(self, dataset_name, all_ious, class_names, sample_count):
        """Analyze IoU results"""
        all_ious = np.array(all_ious)
        mean_ious = np.mean(all_ious, axis=0)
        overall_miou = np.mean(mean_ious)
        lane_miou = np.mean(mean_ious[1:])
        
        print(f"\n{dataset_name} Results:")
        print(f"  Samples tested: {sample_count}")
        print(f"  Overall mIoU: {overall_miou*100:.1f}%")
        print(f"  Lane mIoU: {lane_miou*100:.1f}%")
        
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {mean_ious[i]*100:.1f}%")
        
        return {
            'dataset': dataset_name,
            'samples': sample_count,
            'overall_miou': overall_miou,
            'lane_miou': lane_miou,
            'class_ious': mean_ious,
            'class_names': class_names
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive testing across all datasets"""
        print("COMPREHENSIVE MULTI-DATASET TESTING")
        print("=" * 60)
        
        results = []
        
        # Test 1: AEL Validation (baseline with ground truth)
        ael_result = self.test_ael_validation(max_samples=100)
        results.append(ael_result)
        
        # Test 2: Australian Images (no ground truth)
        australian_result = self.test_australian_images(max_samples=10)
        if australian_result:
            results.append(australian_result)
        
        # Test 3: SS_Dense Dataset
        ss_dense_result = self.test_ss_dataset('SS_Dense', max_samples=30)
        if ss_dense_result:
            results.append(ss_dense_result)
        
        # Test 4: SS_Multi_Lane Dataset  
        ss_multi_result = self.test_ss_dataset('SS_Multi_Lane', max_samples=30)
        if ss_multi_result:
            results.append(ss_multi_result)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        print(f"\n" + "="*60)
        print("MULTI-DATASET PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"{'Dataset':<20} {'Samples':<8} {'Performance':<15} {'Coverage':<12}")
        print("-" * 60)
        
        for result in results:
            dataset = result['dataset']
            samples = result['samples']
            
            if 'overall_miou' in result:
                # Results with ground truth
                performance = f"{result['overall_miou']*100:.1f}% mIoU"
                coverage = f"{result['lane_miou']*100:.1f}% lane"
            else:
                # Results without ground truth
                performance = "N/A (no GT)"
                coverage = f"{result['avg_lane_coverage']*100:.1f}% avg"
            
            print(f"{dataset:<20} {samples:<8} {performance:<15} {coverage:<12}")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        print("-" * 30)
        
        # Find AEL baseline
        ael_result = next((r for r in results if 'AEL' in r['dataset']), None)
        if ael_result:
            print(f"✓ AEL Baseline: {ael_result['overall_miou']*100:.1f}% mIoU")
            
            if ael_result['overall_miou'] >= 0.80:
                print("🏆 TARGET ACHIEVED: ≥80% mIoU on validation set!")
            elif ael_result['overall_miou'] >= 0.78:
                print(f"📈 Close to target: {ael_result['overall_miou']*100:.1f}% (need 80%)")
            else:
                print(f"📊 Below target: {ael_result['overall_miou']*100:.1f}% mIoU")
        
        # Coverage analysis
        coverage_results = [r for r in results if 'avg_lane_coverage' in r]
        if coverage_results:
            avg_coverages = [r['avg_lane_coverage'] for r in coverage_results]
            overall_avg = np.mean(avg_coverages)
            print(f"✓ Average lane detection: {overall_avg*100:.1f}% across datasets")
        
        print("="*60)

def main():
    tester = MultiDatasetTester()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    results = main()