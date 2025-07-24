#!/usr/bin/env python3
"""
Enhanced Post-Processing Pipeline for Lane Detection.
Implements Test-Time Augmentation, Morphological Operations, and CRF-lite.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects
from skimage import measure
import argparse
import logging

logger = logging.getLogger(__name__)

class TestTimeAugmentation:
    """
    Test-Time Augmentation for improved inference performance.
    Applies multiple transformations and averages predictions.
    """
    
    def __init__(self, scales=[0.8, 1.0, 1.2], flips=[False, True]):
        self.scales = scales
        self.flips = flips
        self.augmentations = []
        
        # Generate all combinations
        for scale in scales:
            for flip in flips:
                self.augmentations.append({
                    'scale': scale,
                    'flip': flip
                })
        
        logger.info(f"TTA initialized with {len(self.augmentations)} augmentations")
    
    def apply_augmentation(self, image, aug_params):
        """Apply single augmentation to image."""
        B, C, H, W = image.shape
        
        # Scale
        if aug_params['scale'] != 1.0:
            new_size = int(H * aug_params['scale'])
            image = F.interpolate(image, size=(new_size, new_size), 
                                mode='bilinear', align_corners=False)
            
            # Pad or crop to original size
            if new_size > H:
                # Crop center
                start = (new_size - H) // 2
                image = image[:, :, start:start+H, start:start+W]
            elif new_size < H:
                # Pad
                pad = (H - new_size) // 2
                image = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        
        # Horizontal flip
        if aug_params['flip']:
            image = torch.flip(image, dims=[3])
        
        return image
    
    def reverse_augmentation(self, prediction, aug_params):
        """Reverse augmentation on prediction."""
        # Reverse flip first
        if aug_params['flip']:
            prediction = torch.flip(prediction, dims=[3])
        
        # Scale is handled by interpolation to original size
        return prediction
    
    def __call__(self, model, image):
        """Apply TTA and return averaged prediction."""
        model.eval()
        original_size = image.shape[2:]
        predictions = []
        
        with torch.no_grad():
            for aug_params in self.augmentations:
                # Apply augmentation
                aug_image = self.apply_augmentation(image, aug_params)
                
                # Get prediction
                pred = model(aug_image)
                
                # Resize to original size
                pred = F.interpolate(pred, size=original_size, 
                                   mode='bilinear', align_corners=False)
                
                # Reverse augmentation
                pred = self.reverse_augmentation(pred, aug_params)
                
                predictions.append(pred)
        
        # Average all predictions
        final_prediction = torch.stack(predictions).mean(dim=0)
        return final_prediction

class LanePostProcessor:
    """Enhanced post-processing for lane segmentation"""
    
    def __init__(self, min_lane_area=100, kernel_size=3):
        self.min_lane_area = min_lane_area
        self.kernel_size = kernel_size
        
    def morphological_cleanup(self, predictions):
        """Apply morphological operations to clean up predictions"""
        # Convert to numpy for OpenCV operations
        if torch.is_tensor(predictions):
            pred_np = predictions.cpu().numpy()
        else:
            pred_np = predictions
            
        cleaned = np.zeros_like(pred_np)
        
        # Process each class separately
        for class_id in range(pred_np.max() + 1):
            class_mask = (pred_np == class_id).astype(np.uint8)
            
            if class_id == 0:  # Background - no processing needed
                cleaned[pred_np == class_id] = class_id
                continue
                
            # Morphological opening (remove noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
            
            # Morphological closing (connect broken parts)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # Remove small objects
            closed = remove_small_objects(closed.astype(bool), min_size=self.min_lane_area).astype(np.uint8)
            
            cleaned[closed == 1] = class_id
            
        return torch.tensor(cleaned) if torch.is_tensor(predictions) else cleaned
    
    def enforce_lane_connectivity(self, predictions, max_gap=20):
        """Enforce lane connectivity by filling small gaps"""
        if torch.is_tensor(predictions):
            pred_np = predictions.cpu().numpy()
            return_tensor = True
        else:
            pred_np = predictions
            return_tensor = False
            
        connected = pred_np.copy()
        
        # Process lane classes only (skip background)
        for class_id in range(1, pred_np.max() + 1):
            class_mask = (pred_np == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
                
            # Find skeleton of lane markings
            skeleton = skeletonize(class_mask).astype(np.uint8)
            
            # Dilate skeleton to fill small gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
            dilated = cv2.dilate(skeleton, kernel, iterations=1)
            
            # Intersect with original predictions to avoid over-expansion
            enhanced = cv2.bitwise_and(dilated, class_mask)
            
            # Update connected prediction
            connected[enhanced == 1] = class_id
            
        return torch.tensor(connected) if return_tensor else connected
    
    def apply_crf_lite(self, predictions, original_image, iterations=5):
        """Lightweight CRF-like smoothing using bilateral filtering"""
        if torch.is_tensor(predictions):
            pred_np = predictions.cpu().numpy().astype(np.float32)
            return_tensor = True
        else:
            pred_np = predictions.astype(np.float32)
            return_tensor = False
            
        if torch.is_tensor(original_image):
            img_np = original_image.cpu().numpy()
            if img_np.ndim == 4:  # Batch dimension
                img_np = img_np[0]
            if img_np.shape[0] == 3:  # CHW -> HWC
                img_np = img_np.transpose(1, 2, 0)
        else:
            img_np = original_image
            
        # Ensure image is in correct format
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
            
        # Apply bilateral filtering for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(pred_np, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Convert back to class predictions
        smoothed_classes = np.round(smoothed).astype(pred_np.dtype)
        
        return torch.tensor(smoothed_classes) if return_tensor else smoothed_classes
    
    def full_post_processing(self, predictions, original_image=None):
        """Apply full post-processing pipeline"""
        # Step 1: Morphological cleanup
        cleaned = self.morphological_cleanup(predictions)
        
        # Step 2: Connectivity enforcement
        connected = self.enforce_lane_connectivity(cleaned)
        
        # Step 3: CRF-lite smoothing (if original image available)
        if original_image is not None:
            final = self.apply_crf_lite(connected, original_image)
        else:
            final = connected
            
        return final

class EnhancedPostProcessor:
    """
    Complete enhanced post-processing pipeline.
    Combines TTA, morphological operations, and CRF smoothing.
    """
    
    def __init__(self, use_tta=True, use_morphology=True, use_crf=True):
        self.use_tta = use_tta
        self.use_morphology = use_morphology
        self.use_crf = use_crf
        
        # Initialize components
        if use_tta:
            self.tta = TestTimeAugmentation(scales=[0.9, 1.0, 1.1], flips=[False, True])
        
        if use_morphology:
            self.morphology = LanePostProcessor(min_lane_area=150, kernel_size=3)
        
        logger.info(f"Enhanced post-processor initialized (TTA: {use_tta}, Morph: {use_morphology}, CRF: {use_crf})")
    
    def process(self, model, image, return_intermediate=False):
        """
        Complete enhanced post-processing pipeline.
        Args:
            model: Trained lane detection model
            image: Input image tensor [B, C, H, W]
            return_intermediate: Whether to return intermediate results
        Returns:
            final_prediction: Enhanced prediction
            intermediate_results: Dict of intermediate results (if requested)
        """
        intermediate_results = {}
        
        # Step 1: Get base prediction with TTA
        with torch.no_grad():
            if self.use_tta:
                prediction = self.tta(model, image)
                intermediate_results['tta_prediction'] = prediction.clone()
            else:
                model.eval()
                prediction = model(image)
                intermediate_results['base_prediction'] = prediction.clone()
        
        # Convert to class predictions
        final_prediction = torch.argmax(prediction, dim=1)
        
        # Step 2: Apply morphological and CRF post-processing
        if self.use_morphology or self.use_crf:
            enhanced_predictions = []
            for b in range(final_prediction.shape[0]):
                pred_single = final_prediction[b]
                img_single = image[b] if self.use_crf else None
                
                if self.use_morphology and self.use_crf:
                    enhanced = self.morphology.full_post_processing(pred_single, img_single)
                elif self.use_morphology:
                    enhanced = self.morphology.morphological_cleanup(pred_single)
                    enhanced = self.morphology.enforce_lane_connectivity(enhanced)
                elif self.use_crf:
                    enhanced = self.morphology.apply_crf_lite(pred_single, img_single)
                else:
                    enhanced = pred_single
                
                enhanced_predictions.append(enhanced)
            
            final_prediction = torch.stack(enhanced_predictions).to(final_prediction.device)
            intermediate_results['enhanced_prediction'] = final_prediction.clone()
        
        if return_intermediate:
            return final_prediction, intermediate_results
        else:
            return final_prediction

def test_enhanced_post_processing():
    """Test the enhanced post-processing pipeline."""
    logger.info("Testing Enhanced Post-Processing Pipeline")
    logger.info("=" * 50)
    
    # Create dummy data
    batch_size, channels, height, width = 1, 3, 512, 512
    num_classes = 3
    
    dummy_image = torch.randn(batch_size, channels, height, width)
    dummy_model_output = torch.randn(batch_size, num_classes, height, width)
    
    # Mock model for testing
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return dummy_model_output
    
    mock_model = MockModel()
    
    # Test individual components
    print("Testing TTA...")
    tta = TestTimeAugmentation(scales=[0.9, 1.0], flips=[False, True])
    tta_result = tta(mock_model, dummy_image)
    print(f"TTA output shape: {tta_result.shape}")
    
    print("\nTesting Morphological Processor...")
    morphology = LanePostProcessor()
    dummy_mask = np.random.randint(0, 4, (height, width)).astype(np.uint8)
    cleaned_mask = morphology.morphological_cleanup(dummy_mask)
    print(f"Morphology input shape: {dummy_mask.shape}")
    print(f"Morphology output shape: {cleaned_mask.shape}")
    
    print("\nTesting Complete Pipeline...")
    processor = EnhancedPostProcessor(use_tta=True, use_morphology=True, use_crf=True)
    final_result, intermediate = processor.process(mock_model, dummy_image, return_intermediate=True)
    
    print(f"Final prediction shape: {final_result.shape}")
    print(f"Intermediate results: {list(intermediate.keys())}")
    
    print("\n✅ All enhanced post-processing tests passed!")

def evaluate_post_processing():
    """Evaluate post-processing improvements"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='work_dirs/premium_gpu_best_model.pth')
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()
    
    print("=== Post-Processing Evaluation ===")
    
    # Load model and data (same as TTA script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import and load model
    from premium_gpu_train import PremiumLaneNet, PremiumDataset, calculate_balanced_iou
    
    model = PremiumLaneNet(num_classes=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load validation data
    val_dataset = PremiumDataset("data/ael_mmseg/img_dir/val", "data/ael_mmseg/ann_dir/val", mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize post-processor
    post_processor = LanePostProcessor(min_lane_area=50, kernel_size=3)
    
    # Evaluate
    raw_ious = []
    processed_ious = []
    
    sample_count = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            if sample_count >= args.max_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            raw_pred = torch.argmax(outputs, dim=1)[0]
            
            # Apply post-processing
            processed_pred = post_processor.full_post_processing(raw_pred, images[0])
            
            # Calculate IoUs
            raw_iou, class_names = calculate_balanced_iou(raw_pred, masks[0])
            processed_iou, _ = calculate_balanced_iou(processed_pred, masks[0])
            
            raw_ious.append(raw_iou)
            processed_ious.append(processed_iou)
            
            sample_count += 1
    
    # Results
    raw_mean = np.mean(raw_ious, axis=0)
    processed_mean = np.mean(processed_ious, axis=0)
    
    raw_overall = np.mean(raw_mean)
    processed_overall = np.mean(processed_mean)
    
    print(f"Raw mIoU: {raw_overall:.1%}")
    print(f"Post-processed mIoU: {processed_overall:.1%}")
    print(f"Improvement: +{processed_overall - raw_overall:.1%}")
    
    print("\nPer-class comparison:")
    for i, name in enumerate(class_names):
        improvement = processed_mean[i] - raw_mean[i]
        print(f"{name}: {raw_mean[i]:.1%} → {processed_mean[i]:.1%} (+{improvement:.1%})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_enhanced_post_processing()
    else:
        evaluate_post_processing()