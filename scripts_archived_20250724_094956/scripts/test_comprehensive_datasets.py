#!/usr/bin/env python3
"""
Comprehensive Model Testing - Test best model against multiple datasets
Including: Australian imagery, SS_Dense, SS_Multi_Lane, and original test set
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
import sys

# Add scripts to path
sys.path.append('scripts')
from premium_gpu_train import PremiumLaneNet

def load_model(model_path):
    """Load the best model checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = PremiumLaneNet(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    epoch = checkpoint.get('epoch', 'unknown')
    miou = checkpoint.get('best_miou', 0)
    
    print(f"Model loaded: Epoch {epoch}, mIoU: {miou*100:.1f}%")
    return model, device

def preprocess_image(image_path, target_size=(512, 512)):
    """Preprocess image for model inference"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    original_size = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, (original_size, image_rgb)

def analyze_prediction(pred_mask, class_names=['background', 'white_solid', 'white_dashed']):
    """Analyze prediction mask and extract statistics"""
    total_pixels = pred_mask.size
    stats = {}
    
    for class_id, class_name in enumerate(class_names):
        class_pixels = (pred_mask == class_id).sum()
        percentage = (class_pixels / total_pixels) * 100
        stats[class_name] = {
            'pixels': int(class_pixels),
            'percentage': float(percentage)
        }
    
    return stats

def test_image(model, device, image_path, output_dir, dataset_name):
    """Test a single image and save results"""
    print(f"Testing: {image_path.name}")
    
    # Preprocess
    image_tensor, (original_size, original_rgb) = preprocess_image(image_path)
    if image_tensor is None:
        print(f"  ERROR: Could not load image")
        return None
    
    # Inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Get prediction
        pred_logits = F.softmax(outputs, dim=1)
        pred_mask = torch.argmax(pred_logits, dim=1).cpu().numpy()[0]
        confidence_maps = pred_logits.cpu().numpy()[0]
    
    # Analyze results
    stats = analyze_prediction(pred_mask)
    
    # Calculate detection confidence
    white_solid_conf = confidence_maps[1][pred_mask == 1]
    white_dashed_conf = confidence_maps[2][pred_mask == 2]
    
    confidences = []
    if len(white_solid_conf) > 0:
        confidences.append(np.mean(white_solid_conf))
    if len(white_dashed_conf) > 0:
        confidences.append(np.mean(white_dashed_conf))
    
    lane_confidence = np.mean(confidences) if confidences else 0.0
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name}: {image_path.name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Prediction mask
    colors = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # black, white, yellow
    pred_colored = np.array(colors)[pred_mask]
    axes[0, 1].imshow(pred_colored)
    axes[0, 1].set_title('Lane Detection')
    axes[0, 1].axis('off')
    
    # Overlay
    overlay = original_rgb.copy()
    overlay_resized = cv2.resize(overlay, (512, 512))
    lane_mask = pred_mask > 0
    overlay_resized[lane_mask] = overlay_resized[lane_mask] * 0.6 + pred_colored[lane_mask] * 0.4
    axes[0, 2].imshow(overlay_resized)
    axes[0, 2].set_title('Overlay')
    axes[0, 2].axis('off')
    
    # Confidence maps
    axes[1, 0].imshow(confidence_maps[1], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('White Solid Confidence')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(confidence_maps[2], cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('White Dashed Confidence')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f"""Lane Coverage Statistics:
    
Background: {stats['background']['percentage']:.1f}%
White Solid: {stats['white_solid']['percentage']:.1f}%  
White Dashed: {stats['white_dashed']['percentage']:.1f}%

Total Lane Coverage: {stats['white_solid']['percentage'] + stats['white_dashed']['percentage']:.1f}%
Average Confidence: {lane_confidence:.3f}

Image: {image_path.name}
Dataset: {dataset_name}
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    result_file = output_dir / f"{dataset_name}_{image_path.stem}_analysis.png"
    plt.savefig(result_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return stats
    result = {
        'image': str(image_path),
        'dataset': dataset_name,
        'stats': stats,
        'lane_coverage': stats['white_solid']['percentage'] + stats['white_dashed']['percentage'],
        'confidence': float(lane_confidence),
        'result_file': str(result_file)
    }
    
    print(f"  Lane coverage: {result['lane_coverage']:.1f}%, Confidence: {lane_confidence:.3f}")
    return result

def test_dataset_directory(model, device, dataset_dir, dataset_name, output_dir, image_subdir="images"):
    """Test all images in a dataset directory"""
    print(f"\n=== Testing {dataset_name} Dataset ===")
    
    # Find images
    image_dir = Path(dataset_dir) / "test" / image_subdir
    if not image_dir.exists():
        print(f"Directory not found: {image_dir}")
        return []
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    results = []
    for image_file in sorted(image_files):
        result = test_image(model, device, image_file, output_dir, dataset_name)
        if result:
            results.append(result)
    
    return results

def test_australian_images(model, device, output_dir):
    """Test Australian images"""
    print(f"\n=== Testing Australian Images ===")
    
    # Test both locations
    locations = [
        Path("C:/Users/Admin/Downloads"),
        Path("C:/Users/Admin/LaneSegNet/test_images")
    ]
    
    results = []
    for location in locations:
        if location.exists():
            australian_files = list(location.glob("Australia_*.png"))
            print(f"Found {len(australian_files)} Australian images in {location}")
            
            for image_file in sorted(australian_files):
                result = test_image(model, device, image_file, output_dir, "Australian_Real_World")
                if result:
                    results.append(result)
    
    return results

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL TESTING")
    print("Testing 85.1% mIoU model against multiple datasets")
    print("=" * 80)
    
    # Load best model
    model_path = "work_dirs/premium_gpu_best_model.pth"
    model, device = load_model(model_path)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"comprehensive_test_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    # Test SS_Dense dataset
    ss_dense_results = test_dataset_directory(
        model, device, "data/SS_Dense", "SS_Dense_Munich_Tunnel", output_dir
    )
    all_results.extend(ss_dense_results)
    
    # Test SS_Multi_Lane dataset  
    ss_multi_results = test_dataset_directory(
        model, device, "data/SS_Multi_Lane", "SS_Multi_Lane_Highway", output_dir
    )
    all_results.extend(ss_multi_results)
    
    # Test Australian images
    australian_results = test_australian_images(model, device, output_dir)
    all_results.extend(australian_results)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TESTING SUMMARY")
    print("=" * 80)
    
    # Group by dataset
    dataset_summaries = {}
    for result in all_results:
        dataset = result['dataset']
        if dataset not in dataset_summaries:
            dataset_summaries[dataset] = []
        dataset_summaries[dataset].append(result)
    
    # Print dataset summaries
    for dataset_name, dataset_results in dataset_summaries.items():
        if not dataset_results:
            continue
            
        lane_coverages = [r['lane_coverage'] for r in dataset_results]
        confidences = [r['confidence'] for r in dataset_results]
        
        print(f"\n{dataset_name}:")
        print(f"  Images tested: {len(dataset_results)}")
        print(f"  Average lane coverage: {np.mean(lane_coverages):.1f}%")
        print(f"  Lane coverage range: {np.min(lane_coverages):.1f}% - {np.max(lane_coverages):.1f}%")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
    
    # Save detailed results
    results_file = output_dir / "comprehensive_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': dataset_summaries,
            'detailed_results': all_results,
            'model_info': {
                'path': model_path,
                'timestamp': timestamp
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Visualizations saved in: {output_dir}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()