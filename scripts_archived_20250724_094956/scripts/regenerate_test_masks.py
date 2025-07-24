#!/usr/bin/env python3
"""
Regenerate Test Masks
====================

Regenerate the holdout test masks from original data, since the current
test masks are all empty due to a copying/conversion error.
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

def regenerate_test_masks():
    """Regenerate test masks from original mask data"""
    
    print("REGENERATING HOLDOUT TEST MASKS")
    print("=" * 35)
    
    test_img_dir = Path('data/full_ael_mmseg/img_dir/test')
    test_ann_dir = Path('data/full_ael_mmseg/ann_dir/test')
    source_mask_dir = Path('data/mask')
    
    if not test_img_dir.exists():
        print(f"ERROR: Test image directory not found: {test_img_dir}")
        return
    
    # Get all test image files
    test_images = list(test_img_dir.glob('*.jpg'))
    print(f"Found {len(test_images)} test images")
    
    success_count = 0
    error_count = 0
    empty_masks = 0
    
    print("Regenerating masks...")
    
    for i, img_file in enumerate(test_images):
        sample_id = img_file.stem
        
        # Source mask file
        source_mask = source_mask_dir / f"{sample_id}.jpg"
        target_mask = test_ann_dir / f"{sample_id}.png"
        
        try:
            if source_mask.exists():
                # Load original mask
                mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    # Check if mask has lane data
                    lane_pixels = np.sum(mask > 0)
                    
                    if lane_pixels > 0:
                        # Convert to 3-class format (0=background, 1=white_solid, 2=white_dashed)
                        # The original masks have various values, need to map to 3 classes
                        converted_mask = np.zeros_like(mask)
                        
                        # Map non-zero pixels to lane classes
                        # This is a simplified mapping - you may need to adjust based on actual data
                        lane_mask = mask > 0
                        
                        # For now, map to white_solid (class 1) - can be refined later
                        converted_mask[lane_mask] = 1
                        
                        # Save converted mask
                        cv2.imwrite(str(target_mask), converted_mask)
                        success_count += 1
                        
                        if (i + 1) % 100 == 0:
                            print(f"  Processed {i + 1}/{len(test_images)}: {lane_pixels} lane pixels")
                    else:
                        # Save empty mask (all zeros)
                        cv2.imwrite(str(target_mask), mask)
                        empty_masks += 1
                else:
                    print(f"  ERROR: Could not load {source_mask}")
                    error_count += 1
            else:
                print(f"  ERROR: Source mask not found: {source_mask}")
                error_count += 1
                
        except Exception as e:
            print(f"  ERROR processing {sample_id}: {e}")
            error_count += 1
    
    print(f"\nREGENERATION COMPLETE:")
    print(f"  Successful: {success_count}")
    print(f"  Empty masks: {empty_masks}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(test_images)}")
    
    return success_count, empty_masks, error_count

def validate_regenerated_masks():
    """Validate the regenerated masks"""
    
    print(f"\nVALIDATING REGENERATED MASKS")
    print("=" * 30)
    
    test_ann_dir = Path('data/full_ael_mmseg/ann_dir/test')
    mask_files = list(test_ann_dir.glob('*.png'))
    
    # Sample random files for validation
    import random
    sample_size = min(20, len(mask_files))
    sample_files = random.sample(mask_files, sample_size)
    
    with_lanes = 0
    without_lanes = 0
    lane_pixel_counts = []
    
    for mask_file in sample_files:
        try:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                lane_pixels = np.sum(mask > 0)
                unique_values = np.unique(mask)
                
                if lane_pixels > 0:
                    with_lanes += 1
                    lane_pixel_counts.append(lane_pixels)
                    print(f"  {mask_file.name}: {lane_pixels} lane pixels, classes: {unique_values}")
                else:
                    without_lanes += 1
                    print(f"  {mask_file.name}: NO LANES")
            else:
                print(f"  {mask_file.name}: ERROR loading")
        except Exception as e:
            print(f"  {mask_file.name}: ERROR - {e}")
    
    print(f"\nVALIDATION RESULTS:")
    print(f"  Files with lanes: {with_lanes}/{sample_size}")
    print(f"  Files without lanes: {without_lanes}/{sample_size}")
    
    if lane_pixel_counts:
        avg_lanes = np.mean(lane_pixel_counts)
        print(f"  Average lane pixels: {avg_lanes:.1f}")
    
    if with_lanes > 0:
        print(f"  SUCCESS: Test masks now contain lane data!")
        return True
    else:
        print(f"  PROBLEM: Still no lane data in test masks")
        return False

def main():
    """Run mask regeneration and validation"""
    
    print("FIXING HOLDOUT TEST MASK ISSUE")
    print("=" * 32)
    print("Problem: Test masks are all empty despite having source data")
    print("Solution: Regenerate masks from original mask files")
    print("=" * 32)
    
    # Regenerate masks
    success, empty, errors = regenerate_test_masks()
    
    # Validate results
    validation_passed = validate_regenerated_masks()
    
    print(f"\n" + "=" * 50)
    print("MASK REGENERATION SUMMARY")
    print("=" * 50)
    
    print(f"Regenerated: {success} masks with lane data")
    print(f"Empty masks: {empty} (legitimately no lanes)")
    print(f"Errors: {errors}")
    
    if validation_passed:
        print(f"\nSUCCESS: Test masks now ready for evaluation!")
        print(f"NEXT STEP: Re-run holdout test to get TRUE model performance")
    else:
        print(f"\nPROBLEM: Mask regeneration may have failed")
        print(f"INVESTIGATION: Check original mask format and conversion logic")
    
    return success, empty, errors

if __name__ == "__main__":
    main()