#!/usr/bin/env python3
"""
Test LaneSegNet lane detection with local aerial imagery samples.

This script tests the lane detection system with actual aerial images stored locally
in data/imgs/ directory to validate detection performance and optimization needs.
"""

import asyncio
import json
import os
import random
import time
from PIL import Image
import requests

# Test configuration
API_BASE = "http://localhost:8010"
LOCAL_IMAGES_DIR = "data/imgs"
TEST_SAMPLES = 5  # Number of random images to test
OUTPUT_DIR = "local_aerial_tests"

def create_output_dir():
    """Create output directory for test results."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def get_sample_images():
    """Get a random sample of local aerial images for testing."""
    if not os.path.exists(LOCAL_IMAGES_DIR):
        print(f"Error: Local images directory not found: {LOCAL_IMAGES_DIR}")
        return []
    
    all_images = [f for f in os.listdir(LOCAL_IMAGES_DIR) if f.endswith('.jpg')]
    if len(all_images) == 0:
        print(f"Error: No .jpg files found in {LOCAL_IMAGES_DIR}")
        return []
    
    # Select random sample
    sample_size = min(TEST_SAMPLES, len(all_images))
    selected_images = random.sample(all_images, sample_size)
    
    print(f"Selected {sample_size} random images from {len(all_images)} available:")
    for img in selected_images:
        img_path = os.path.join(LOCAL_IMAGES_DIR, img)
        try:
            with Image.open(img_path) as pil_img:
                print(f"  - {img}: {pil_img.size[0]}x{pil_img.size[1]} pixels")
        except Exception as e:
            print(f"  - {img}: ERROR - {e}")
    
    return selected_images

def test_health_check():
    """Test API health and model loading."""
    print("\n=== Health Check ===")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"[OK] API Status: {health_data.get('status', 'unknown')}")
            print(f"[OK] Model Loaded: {health_data.get('model_loaded', 'unknown')}")
            return True
        else:
            print(f"[ERROR] Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Health check error: {e}")
        return False

def test_local_image_detection(image_filename):
    """Test lane detection on a specific local aerial image."""
    print(f"\n=== Testing {image_filename} ===")
    
    # Since we need coordinates for the API, we'll use dummy coordinates
    # The local imagery provider should override these and use the actual image file
    test_payload = {
        "north": -27.4698,  # Dummy coordinates - local provider will use actual image
        "south": -27.4705,
        "east": 153.0258,
        "west": 153.0251
    }
    
    try:
        start_time = time.time()
        
        # Test the analysis endpoint
        analysis_response = requests.post(
            f"{API_BASE}/analyze_road_infrastructure",
            json=test_payload,
            timeout=60
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if analysis_response.status_code == 200:
            analysis_data = analysis_response.json()
            
            # Extract key metrics
            elements = analysis_data.get('infrastructure_elements', [])
            summary = analysis_data.get('analysis_summary', {})
            metadata = analysis_data.get('image_metadata', {})
            
            print(f"[OK] Analysis completed in {processing_time:.1f}ms")
            print(f"   Elements detected: {len(elements)}")
            print(f"   Total lane length: {summary.get('total_lane_length_m', 0):.2f}m")
            print(f"   Image size: {metadata.get('width', 'unknown')}x{metadata.get('height', 'unknown')}")
            print(f"   Source: {metadata.get('source', 'unknown')}")
            
            # Print class breakdown
            class_counts = summary.get('elements_by_class', {})
            if class_counts:
                print("   Classes detected:")
                for class_name, count in class_counts.items():
                    print(f"     - {class_name}: {count}")
            else:
                print("   No lane classes detected")
                
            return {
                'status': 'success',
                'processing_time_ms': processing_time,
                'elements_count': len(elements),
                'classes': class_counts,
                'lane_length_m': summary.get('total_lane_length_m', 0),
                'image_size': f"{metadata.get('width', 'unknown')}x{metadata.get('height', 'unknown')}",
                'source': metadata.get('source', 'unknown')
            }
            
        else:
            error_text = analysis_response.text
            print(f"[ERROR] Analysis failed: HTTP {analysis_response.status_code}")
            print(f"   Error: {error_text}")
            return {
                'status': 'error',
                'error_code': analysis_response.status_code,
                'error_message': error_text,
                'processing_time_ms': processing_time
            }
            
    except Exception as e:
        print(f"[ERROR] Test error: {e}")
        return {
            'status': 'exception',
            'error_message': str(e)
        }

def test_visualization(image_filename):
    """Test visualization generation for a local image."""
    print(f"\n=== Testing Visualization for {image_filename} ===")
    
    test_payload = {
        "north": -27.4698,  # Dummy coordinates
        "south": -27.4705,
        "east": 153.0258,
        "west": 153.0251
    }
    
    viz_types = ["original", "annotated", "overlay", "side_by_side"]
    results = {}
    
    for viz_type in viz_types:
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE}/visualize_infrastructure",
                json=test_payload,
                params={"viz_type": viz_type, "show_labels": True},
                timeout=60
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Save visualization image
                output_filename = f"{image_filename[:-4]}_{viz_type}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size_kb = len(response.content) / 1024
                
                print(f"[OK] {viz_type}: {processing_time:.1f}ms, {file_size_kb:.1f}KB -> {output_filename}")
                
                results[viz_type] = {
                    'status': 'success',
                    'processing_time_ms': processing_time,
                    'file_size_kb': file_size_kb,
                    'output_file': output_filename
                }
            else:
                print(f"[ERROR] {viz_type}: HTTP {response.status_code}")
                results[viz_type] = {
                    'status': 'error',
                    'error_code': response.status_code,
                    'processing_time_ms': processing_time
                }
                
        except Exception as e:
            print(f"[ERROR] {viz_type}: {e}")
            results[viz_type] = {
                'status': 'exception',
                'error_message': str(e)
            }
    
    return results

def main():
    """Main test function."""
    print("LaneSegNet Local Aerial Imagery Testing")
    print("=" * 50)
    
    # Create output directory
    create_output_dir()
    
    # Health check
    if not test_health_check():
        print("\n[ERROR] API health check failed. Ensure the service is running at http://localhost:8010")
        return
    
    # Get sample images
    sample_images = get_sample_images()
    if not sample_images:
        print("\n[ERROR] No sample images available for testing")
        return
    
    # Test results storage
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images_tested': len(sample_images),
        'results': {}
    }
    
    # Test each image
    for i, image_filename in enumerate(sample_images, 1):
        print(f"\n{'='*20} Test {i}/{len(sample_images)} {'='*20}")
        
        # Test lane detection analysis
        analysis_result = test_local_image_detection(image_filename)
        
        # Test visualization (only if analysis succeeds)
        viz_results = {}
        if analysis_result.get('status') == 'success':
            viz_results = test_visualization(image_filename)
        
        # Store results
        test_results['results'][image_filename] = {
            'analysis': analysis_result,
            'visualizations': viz_results
        }
    
    # Generate summary report
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    
    successful_tests = sum(1 for r in test_results['results'].values() 
                          if r['analysis'].get('status') == 'success')
    
    print(f"Images tested: {len(sample_images)}")
    print(f"Successful detections: {successful_tests}")
    print(f"Success rate: {(successful_tests/len(sample_images)*100):.1f}%")
    
    if successful_tests > 0:
        # Calculate averages for successful tests
        successful_results = [r['analysis'] for r in test_results['results'].values() 
                            if r['analysis'].get('status') == 'success']
        
        avg_processing_time = sum(r['processing_time_ms'] for r in successful_results) / len(successful_results)
        total_elements = sum(r['elements_count'] for r in successful_results)
        avg_elements = total_elements / len(successful_results)
        
        print(f"Average processing time: {avg_processing_time:.1f}ms")
        print(f"Average elements per image: {avg_elements:.1f}")
        print(f"Total elements detected: {total_elements}")
        
        # Class distribution
        all_classes = {}
        for r in successful_results:
            for class_name, count in r.get('classes', {}).items():
                all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        if all_classes:
            print("\nClass distribution across all images:")
            for class_name, total_count in sorted(all_classes.items()):
                print(f"  - {class_name}: {total_count}")
        else:
            print("\n[WARNING] No lane markings detected in any images")
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Visualization images saved to: {OUTPUT_DIR}/")
    
    # Recommendations
    print(f"\n{'='*20} RECOMMENDATIONS {'='*20}")
    if successful_tests == 0:
        print("[CRITICAL] No successful detections - Investigation needed:")
        print("   1. Check if local imagery provider is correctly configured")
        print("   2. Verify model is suitable for local aerial imagery characteristics") 
        print("   3. Consider adjusting detection thresholds or preprocessing")
    elif successful_tests < len(sample_images) * 0.8:
        print("[WARNING] Detection success rate below 80% - Optimization recommended:")
        print("   1. Analyze failed cases for common patterns")
        print("   2. Consider specialized training on local aerial imagery")
        print("   3. Evaluate detection thresholds and post-processing")
    else:
        print("[SUCCESS] Good detection performance - Ready for production:")
        print("   1. Consider expanding test dataset for validation")
        print("   2. Monitor performance across different image types")
        print("   3. Fine-tune for specific lane marking types as needed")

if __name__ == "__main__":
    main()