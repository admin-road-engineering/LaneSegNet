#!/usr/bin/env python3
"""
Test the 85.1% mIoU model on custom aerial images
Upload your images to test/ directory and see the results
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
sys.path.append('scripts')

# Import model architecture
from premium_gpu_train import PremiumLaneNet

def load_85_1_model():
    """Load the 85.1% mIoU model"""
    print("Loading 85.1% mIoU model...")
    
    checkpoint_path = Path('work_dirs/premium_gpu_best_model.pth')
    if not checkpoint_path.exists():
        print("ERROR: 85.1% checkpoint not found!")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same architecture
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"SUCCESS: Loaded 85.1% model on {device}")
    print(f"Model from Epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best mIoU: {checkpoint.get('best_miou', 0)*100:.1f}%")
    print()
    
    return model, device

def preprocess_image(image_path, target_size=512):
    """Preprocess image for model input"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()
    
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (target_size, target_size))
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_image

def predict_lanes(model, image_tensor, device):
    """Run inference on the image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get class predictions
        predictions = torch.argmax(probs, dim=1)
        
        return predictions.cpu().numpy(), probs.cpu().numpy()

def visualize_results(original_image, predictions, probabilities, save_path):
    """Create visualization of results"""
    pred_mask = predictions[0]  # Remove batch dimension
    
    # Create colored mask
    # Class 0: Background (black)
    # Class 1: White solid lanes (white)
    # Class 2: White dashed lanes (yellow)
    
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Background stays black (0, 0, 0)
    
    # White solid lanes (white)
    colored_mask[pred_mask == 1] = [255, 255, 255]
    
    # White dashed lanes (yellow for distinction)
    colored_mask[pred_mask == 2] = [255, 255, 0]
    
    # Resize original image to match prediction size
    original_resized = cv2.resize(original_image, (pred_mask.shape[1], pred_mask.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(original_resized, 0.7, colored_mask, 0.3, 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'85.1% mIoU Model Results - {Path(save_path).stem}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Predictions mask
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('Lane Detection\n(White=Solid, Yellow=Dashed)')
    axes[0, 1].axis('off')
    
    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (Original + Predictions)')
    axes[1, 0].axis('off')
    
    # Confidence heatmap for lane classes (combine class 1 and 2)
    lane_confidence = probabilities[0, 1, :, :] + probabilities[0, 2, :, :]
    im = axes[1, 1].imshow(lane_confidence, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Lane Confidence Heatmap')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Save results
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    total_pixels = pred_mask.size
    background_pixels = np.sum(pred_mask == 0)
    white_solid_pixels = np.sum(pred_mask == 1) 
    white_dashed_pixels = np.sum(pred_mask == 2)
    
    stats = {
        'total_pixels': int(total_pixels),
        'background_pixels': int(background_pixels),
        'white_solid_pixels': int(white_solid_pixels),
        'white_dashed_pixels': int(white_dashed_pixels),
        'background_percentage': float(background_pixels / total_pixels * 100),
        'white_solid_percentage': float(white_solid_pixels / total_pixels * 100),
        'white_dashed_percentage': float(white_dashed_pixels / total_pixels * 100),
        'total_lane_percentage': float((white_solid_pixels + white_dashed_pixels) / total_pixels * 100),
        'max_lane_confidence': float(np.max(lane_confidence)),
        'mean_lane_confidence': float(np.mean(lane_confidence)),
        'lane_pixels_detected': int(white_solid_pixels + white_dashed_pixels)
    }
    
    return stats

def test_85_1_model():
    """Main testing function"""
    print("=" * 70)
    print("TESTING 85.1% mIoU MODEL ON CUSTOM IMAGES")
    print("=" * 70)
    print()
    
    # Create test directory if it doesn't exist
    test_dir = Path('test_images')
    test_dir.mkdir(exist_ok=True)
    
    # Create results directory
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    
    print(f"Instructions:")
    print(f"1. Place your aerial images in: {test_dir}/")
    print(f"2. Supported formats: .jpg, .jpeg, .png")
    print(f"3. Results will be saved in: {results_dir}/")
    print()
    
    # Load model
    model, device = load_85_1_model()
    if model is None:
        return
    
    # Find test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_dir.glob(f'*{ext}')))
        test_images.extend(list(test_dir.glob(f'*{ext.upper()}')))
    
    if not test_images:
        print(f"No images found in {test_dir}/")
        print("Please add some aerial images to test!")
        return
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")
    print()
    
    # Process each image
    all_results = {}
    
    for i, image_path in enumerate(test_images, 1):
        print(f"Processing {i}/{len(test_images)}: {image_path.name}")
        
        # Preprocess
        image_tensor, original_image = preprocess_image(image_path)
        if image_tensor is None:
            print(f"  ERROR: Could not load {image_path.name}")
            continue
        
        # Predict
        predictions, probabilities = predict_lanes(model, image_tensor, device)
        
        # Visualize and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = f"85_1_result_{image_path.stem}_{timestamp}"
        
        # Save visualization
        viz_path = results_dir / f"{result_name}.png"
        stats = visualize_results(original_image, predictions, probabilities, viz_path)
        
        # Save raw prediction
        pred_path = results_dir / f"{result_name}_prediction.npy"
        np.save(pred_path, predictions[0])
        
        # Save statistics
        stats['image_name'] = image_path.name
        stats['model_info'] = '85.1% mIoU Premium Model'
        stats['timestamp'] = timestamp
        
        stats_path = results_dir / f"{result_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        all_results[image_path.name] = stats
        
        print(f"  SUCCESS: Results saved")
        print(f"    Visualization: {viz_path}")
        print(f"    Lane coverage: {stats['total_lane_percentage']:.1f}%")
        print(f"    White solid: {stats['white_solid_percentage']:.1f}%")
        print(f"    White dashed: {stats['white_dashed_percentage']:.1f}%")
        print()
    
    # Save summary
    summary_path = results_dir / f"85_1_model_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 70)
    print("TESTING COMPLETE!")
    print(f"Results saved in: {results_dir}/")
    print(f"Summary: {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    test_85_1_model()