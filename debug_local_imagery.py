#!/usr/bin/env python3
"""
Debug script to test local imagery provider directly.
"""

import os
import sys
import logging
from PIL import Image
import numpy as np

# Add the app directory to the path
sys.path.append('app')

from app.imagery_acquisition import LocalImageryProvider, ImageryAcquisitionManager
from app.schemas import GeographicBounds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_local_provider_direct():
    """Test LocalImageryProvider directly."""
    print("=== Testing LocalImageryProvider Directly ===")
    
    local_dir = 'data/imgs'
    if not os.path.exists(local_dir):
        print(f"ERROR: Local directory does not exist: {local_dir}")
        return False
    
    print(f"Local directory exists: {local_dir}")
    
    # Count images
    image_files = [f for f in os.listdir(local_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} .jpg files")
    
    if len(image_files) == 0:
        print("ERROR: No .jpg files found")
        return False
    
    # Show some examples
    print("First 5 images:")
    for i, img in enumerate(image_files[:5]):
        print(f"  {i+1}. {img}")
    
    # Test LocalImageryProvider
    try:
        provider = LocalImageryProvider(local_dir)
        print("LocalImageryProvider created successfully")
        
        # Test bounds (dummy coordinates)
        bounds = GeographicBounds(north=-27.4698, south=-27.4705, east=153.0258, west=153.0251)
        
        # Try to fetch imagery
        image_array, metadata = provider.fetch_imagery(bounds, 0.1)
        
        print(f"SUCCESS: Fetched image with shape {image_array.shape}")
        print(f"Metadata: {metadata}")
        
        # Save test image
        if len(image_array.shape) == 3:
            test_image = Image.fromarray(image_array)
            test_image.save('debug_local_test.jpg')
            print("Test image saved as debug_local_test.jpg")
        
        return True
        
    except Exception as e:
        print(f"ERROR in LocalImageryProvider: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imagery_manager():
    """Test ImageryAcquisitionManager with local provider."""
    print("\n=== Testing ImageryAcquisitionManager with Local Provider ===")
    
    try:
        manager = ImageryAcquisitionManager()
        print(f"Available providers: {list(manager.providers.keys())}")
        
        # Check if local provider is available
        if 'local' not in manager.providers:
            print("ERROR: Local provider not available in ImageryAcquisitionManager")
            return False
        
        print("Local provider is available")
        
        # Test bounds
        bounds = GeographicBounds(north=-27.4698, south=-27.4705, east=153.0258, west=153.0251)
        
        # Try to fetch with preferred_provider="local"
        print("Attempting to fetch with preferred_provider='local'...")
        image_array, metadata = manager.fetch_best_imagery(bounds, 0.1, preferred_provider="local")
        
        print(f"SUCCESS: Fetched image with shape {image_array.shape}")
        print(f"Source: {metadata.get('source', 'unknown')}")
        
        # Check if it's actually from local provider
        if 'Local' in metadata.get('source', ''):
            print("✓ Successfully used local provider")
            return True
        else:
            print("✗ Used different provider despite preference")
            return False
            
    except Exception as e:
        print(f"ERROR in ImageryAcquisitionManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("Debugging Local Imagery Provider")
    print("=" * 50)
    
    # Test 1: Direct LocalImageryProvider
    success1 = test_local_provider_direct()
    
    # Test 2: ImageryManager
    success2 = test_imagery_manager()
    
    print(f"\n{'='*20} RESULTS {'='*20}")
    print(f"LocalImageryProvider Direct: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"ImageryAcquisitionManager:    {'✓ PASS' if success2 else '✗ FAIL'}")
    
    if success1 and success2:
        print("\n✓ Local imagery provider is working correctly")
        print("The issue may be in the API endpoint configuration")
    elif success1:
        print("\n⚠️ LocalImageryProvider works but ImageryAcquisitionManager has issues")
        print("Check ImageryAcquisitionManager setup and provider registration")
    else:
        print("\n✗ LocalImageryProvider has fundamental issues")
        print("Check directory path and file access permissions")

if __name__ == "__main__":
    main()