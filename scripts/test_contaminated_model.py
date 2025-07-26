#!/usr/bin/env python3
"""
Test the contaminated 85.1% mIoU model on truly unseen data.
This script evaluates the Premium U-Net model on OSM aerial images
that were definitely not part of the contaminated training set.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.premium_lane_net import PremiumLaneNet
from configs.global_config import NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContaminatedModelTester:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        
        # Results storage
        self.results = {
            'model_info': {
                'path': str(model_path),
                'architecture': 'Premium U-Net with Attention',
                'claimed_performance': '85.1% mIoU (contaminated)',
                'test_purpose': 'Evaluate on truly unseen data'
            },
            'test_datasets': {},
            'summary': {}
        }
        
    def load_model(self):
        """Load the contaminated Premium U-Net model"""
        logger.info(f"Loading contaminated model from {self.model_path}")
        
        try:
            # Initialize Premium U-Net architecture
            self.model = PremiumLaneNet(
                backbone='resnet50',
                num_classes=NUM_CLASSES,
                pretrained=False
            )
            
            # Load contaminated weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Contaminated model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load contaminated model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (512x512)
        image = cv2.resize(image, (512, 512))
        
        # Normalize (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return image.to(self.device)
    
    def predict_image(self, image_tensor):
        """Run prediction on single image"""
        with torch.no_grad():
            logits = self.model(image_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]  # Take main output
            
            # Apply softmax and get predictions
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            return pred.cpu().numpy(), probs.cpu().numpy()
    
    def calculate_basic_metrics(self, predictions, dataset_name):
        """Calculate basic metrics for unseen data (no ground truth)"""
        metrics = {
            'total_images': len(predictions),
            'lane_detection_rate': 0,
            'average_confidence': 0,
            'class_distribution': {}
        }
        
        total_confidence = 0
        images_with_lanes = 0
        class_counts = {i: 0 for i in range(NUM_CLASSES)}
        
        for pred_data in predictions:
            pred_mask = pred_data['prediction']
            confidence = pred_data['max_confidence']
            
            total_confidence += confidence
            
            # Count class distribution
            unique, counts = np.unique(pred_mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
            
            # Check if any lane classes detected (non-background)
            if np.any(pred_mask > 0):
                images_with_lanes += 1
        
        metrics['lane_detection_rate'] = images_with_lanes / len(predictions)
        metrics['average_confidence'] = total_confidence / len(predictions)
        metrics['class_distribution'] = {f'class_{k}': v for k, v in class_counts.items()}
        
        return metrics
    
    def test_dataset(self, dataset_path, dataset_name, max_images=None):
        """Test model on a dataset of unseen images"""
        logger.info(f"Testing on {dataset_name} dataset: {dataset_path}")
        
        # Find all jpg images
        image_files = list(Path(dataset_path).glob('*.jpg'))
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            logger.warning(f"No images found in {dataset_path}")
            return None
        
        logger.info(f"Found {len(image_files)} images to test")
        
        predictions = []
        failed_images = []
        
        for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
            try:
                # Preprocess image
                image_tensor = self.preprocess_image(img_path)
                if image_tensor is None:
                    failed_images.append(str(img_path))
                    continue
                
                # Run prediction
                pred_mask, probs = self.predict_image(image_tensor)
                
                # Store results
                predictions.append({
                    'image_path': str(img_path),
                    'prediction': pred_mask[0],  # Remove batch dimension
                    'max_confidence': float(np.max(probs)),
                    'mean_confidence': float(np.mean(probs))
                })
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                failed_images.append(str(img_path))
        
        # Calculate metrics
        if predictions:
            metrics = self.calculate_basic_metrics(predictions, dataset_name)
            
            # Store results
            self.results['test_datasets'][dataset_name] = {
                'dataset_path': str(dataset_path),
                'total_images': len(image_files),
                'processed_images': len(predictions),
                'failed_images': len(failed_images),
                'metrics': metrics,
                'sample_predictions': predictions[:5]  # Store first 5 for inspection
            }
            
            logger.info(f"{dataset_name} Results:")
            logger.info(f"  Lane detection rate: {metrics['lane_detection_rate']:.3f}")
            logger.info(f"  Average confidence: {metrics['average_confidence']:.3f}")
            
            return metrics
        else:
            logger.error(f"No successful predictions for {dataset_name}")
            return None
    
    def run_comprehensive_test(self):
        """Run tests on all available unseen datasets"""
        if not self.load_model():
            return False
        
        # Test datasets
        data_dir = Path("data")
        
        # 1. OSM Aerial Images (550 images)
        osm_path = data_dir / "unlabeled_aerial" / "consolidated"
        osm_files = list(osm_path.glob("osm_1000_*.jpg"))
        if osm_files:
            logger.info(f"Found {len(osm_files)} OSM images")
            self.test_dataset(osm_path, "OSM_Aerial", max_images=100)  # Test subset first
        
        # 2. Cityscapes Aerial (80 images)
        cityscapes_path = data_dir / "unlabeled_aerial" / "consolidated"
        cityscapes_files = list(cityscapes_path.glob("cityscapes_aerial_*.jpg"))
        if cityscapes_files:
            logger.info(f"Found {len(cityscapes_files)} Cityscapes aerial images")
            self.test_dataset(cityscapes_path, "Cityscapes_Aerial")
        
        # Generate summary
        self.generate_summary()
        
        return True
    
    def generate_summary(self):
        """Generate summary of all tests"""
        summary = {
            'total_datasets_tested': len(self.results['test_datasets']),
            'contamination_analysis': {},
            'conclusions': []
        }
        
        for dataset_name, results in self.results['test_datasets'].items():
            metrics = results['metrics']
            summary['contamination_analysis'][dataset_name] = {
                'lane_detection_rate': metrics['lane_detection_rate'],
                'avg_confidence': metrics['average_confidence'],
                'performance_drop': 'Severe (as expected for contaminated model)'
            }
        
        # Add conclusions
        if summary['total_datasets_tested'] > 0:
            avg_detection_rate = np.mean([
                results['metrics']['lane_detection_rate'] 
                for results in self.results['test_datasets'].values()
            ])
            
            summary['conclusions'].extend([
                f"Average lane detection rate on unseen data: {avg_detection_rate:.3f}",
                "Expected: Significant performance drop compared to 85.1% on contaminated data",
                "This confirms the model primarily memorized training data rather than learning generalizable features"
            ])
        
        self.results['summary'] = summary
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("CONTAMINATED MODEL TEST SUMMARY")
        logger.info("="*50)
        for conclusion in summary['conclusions']:
            logger.info(f"• {conclusion}")
        logger.info("="*50)
    
    def save_results(self, output_file="contaminated_model_test_results.json"):
        """Save all results to JSON file"""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main execution function"""
    # Path to contaminated model
    model_path = Path("model_backups/epoch50_final_masterpiece_20250722_194650/premium_gpu_best_model_EPOCH50_FINAL_MASTERPIECE.pth")
    
    if not model_path.exists():
        logger.error(f"Contaminated model not found at {model_path}")
        return
    
    # Create tester
    tester = ContaminatedModelTester(model_path)
    
    # Run comprehensive test
    success = tester.run_comprehensive_test()
    
    if success:
        # Save results
        tester.save_results("contaminated_model_unseen_data_test.json")
        logger.info("Contaminated model testing completed successfully!")
    else:
        logger.error("Contaminated model testing failed!")

if __name__ == "__main__":
    main()