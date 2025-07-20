import json
import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# --- Configuration ---

# 1. Define your class mapping
# YOU MUST REVIEW YOUR JSON ANNOTATIONS TO IDENTIFY ALL UNIQUE COMBINATIONS
# AND ASSIGN THEM APPROPRIATE CLASS IDS.
# Example:
CLASS_MAPPING = {
    # (is_single, is_white, is_solid) : class_id
    (True, True, True): 1,   # Single White Solid
    (True, True, False): 2,  # Single White Dashed
    # Add more combinations as they exist in your data.
    # For example, if you have double lines, you'll need a 'double' property or similar.
    # If you have yellow lines, the 'is_white' would be False, and you'd need another property for yellow.
    # (False, True, True): 3, # e.g. Double White Solid (if 'single=False' means double)
}
CLASS_NAMES = [
    "background", # Class ID 0
    "single_white_solid", # Class ID 1
    "single_white_dashed", # Class ID 2
    # Add other class names corresponding to CLASS_MAPPING
]
DEFAULT_CLASS_ID = 0 # Background for any unmapped lanes

# 2. Define Paths
# Assumes this script is in the root of your LaneSegNet project
# and 'data' is a subdirectory.
BASE_DATA_DIR = Path("data")
OUTPUT_MMSEG_DIR = BASE_DATA_DIR / "ael_mmseg"
IMG_SUBDIR_MMSEG = OUTPUT_MMSEG_DIR / "img_dir"
ANN_SUBDIR_MMSEG = OUTPUT_MMSEG_DIR / "ann_dir"

# 3. Drawing parameters
LANE_THICKNESS = 3 # Thickness of the lane lines drawn on the mask in pixels. Adjust as needed.

# --- Helper Functions ---

def get_class_id(lane_properties):
    """Determines class ID based on lane properties."""
    is_single = lane_properties.get("single", False) # Default to False if missing
    is_white = lane_properties.get("white", False)   # Default to False if missing
    is_solid = lane_properties.get("solid", False)   # Default to False if missing

    key = (is_single, is_white, is_solid)
    return CLASS_MAPPING.get(key, DEFAULT_CLASS_ID)

def create_mask_for_image(image_filename_id, per_image_json_path, output_mask_path):
    """
    Generates a semantic segmentation mask for a single image.
    """
    try:
        with open(per_image_json_path, 'r') as f:
            annotation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation JSON not found: {per_image_json_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON: {per_image_json_path}")
        return False

    resolution = annotation_data.get("resolution")
    if not resolution or len(resolution) != 2:
        print(f"Error: Valid 'resolution' not found in {per_image_json_path}")
        return False

    width, height = resolution
    mask = np.zeros((height, width), dtype=np.uint8) # Initialize with background class (0)

    lanes = annotation_data.get("lanes", [])
    if not lanes:
        print(f"Warning: No lanes found in {per_image_json_path}")
        # Save empty mask
        cv2.imwrite(str(output_mask_path), mask)
        return True

    for lane_idx, lane in enumerate(lanes):
        properties = {
            "single": lane.get("single"),
            "white": lane.get("white"),
            "solid": lane.get("solid")
        }
        class_id = get_class_id(properties)
        
        vertices = lane.get("vertices")
        if not vertices or len(vertices) < 2:
            continue

        # Convert to NumPy array for OpenCV
        try:
            lane_points = np.array(vertices, dtype=np.int32)
        except ValueError:
            print(f"Warning: Could not convert vertices to int32 array in {image_filename_id}")
            continue
            
        # Draw the lane with a thicker line for visibility
        cv2.polylines(
            mask,
            [lane_points],
            isClosed=False,
            color=int(class_id),
            thickness=LANE_THICKNESS
        )

    try:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mask and verify it was written correctly
        success = cv2.imwrite(str(output_mask_path), mask)
        if not success:
            print(f"Error: Failed to write mask: {output_mask_path}")
            return False
            
        # Read back the mask to verify
        saved_mask = cv2.imread(str(output_mask_path), cv2.IMREAD_GRAYSCALE)
        if saved_mask is None:
            print(f"Error: Could not read back saved mask: {output_mask_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error writing mask {output_mask_path}: {e}")
        return False

def process_dataset_split(original_manifest_json_path, split_name, max_images=None):
    """
    Processes a dataset split (train, val) based on its manifest JSON.
    max_images: If set, only process this many images (for testing)
    """
    print(f"\n--- Processing {split_name} data ---")
    manifest_path = BASE_DATA_DIR / original_manifest_json_path
    
    output_img_dir_split = IMG_SUBDIR_MMSEG / split_name
    output_ann_dir_split = ANN_SUBDIR_MMSEG / split_name
    
    output_img_dir_split.mkdir(parents=True, exist_ok=True)
    output_ann_dir_split.mkdir(parents=True, exist_ok=True)

    try:
        with open(manifest_path, 'r') as f:
            data_entries = json.load(f)
    except FileNotFoundError:
        print(f"Error: Manifest file not found: {manifest_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode manifest JSON: {manifest_path}")
        return

    if "data" not in data_entries or not isinstance(data_entries["data"], list):
        print(f"Error: Manifest JSON {manifest_path} does not have the expected 'data' list structure.")
        return
        
    entries = data_entries["data"]
    if max_images:
        entries = entries[:max_images]
        
    total_entries = len(entries)
    print(f"Found {total_entries} entries in {original_manifest_json_path}")
    
    processed_count = 0
    error_count = 0
    
    for i, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 3:
            print(f"Warning: Skipping invalid entry in {manifest_path}: {entry}")
            error_count += 1
            continue

        # Extract filenames from paths
        img_filename = Path(entry[0]).name
        json_annot_filename = Path(entry[1]).name

        # Construct paths relative to our project structure
        current_img_path_in_data = BASE_DATA_DIR / "imgs" / img_filename
        current_json_annot_path_in_data = BASE_DATA_DIR / "json" / json_annot_filename

        output_img_path = output_img_dir_split / img_filename
        output_mask_path = output_ann_dir_split / f"{img_filename.split('.')[0]}.png"

        if not current_img_path_in_data.exists():
            print(f"Warning: Source image not found: {current_img_path_in_data}")
            error_count += 1
            continue
        
        if not current_json_annot_path_in_data.exists():
            print(f"Warning: Source JSON annotation not found: {current_json_annot_path_in_data}")
            error_count += 1
            continue

        # Print progress every 100 images
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{total_entries} ({(i+1)/total_entries*100:.1f}%)")

        # 1. Copy original image
        try:
            shutil.copy(current_img_path_in_data, output_img_path)
        except Exception as e:
            print(f"Error copying image {current_img_path_in_data}: {e}")
            error_count += 1
            continue

        # 2. Create and save the new mask from per-image JSON
        if not create_mask_for_image(img_filename, current_json_annot_path_in_data, output_mask_path):
            print(f"Failed to create mask for {img_filename}")
            if output_img_path.exists():
                output_img_path.unlink()
            error_count += 1
        else:
            processed_count += 1

    print(f"\nFinished processing {split_name} data:")
    print(f"Successfully processed: {processed_count}/{total_entries}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    print("Starting AEL Dataset Mask Generation for MMSegmentation...")
    print(f"Using CLASS_MAPPING: {CLASS_MAPPING}")
    print(f"Using CLASS_NAMES: {CLASS_NAMES}")
    print(f"Lane thickness: {LANE_THICKNESS} pixels")

    # Create base output directories
    OUTPUT_MMSEG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_SUBDIR_MMSEG.mkdir(parents=True, exist_ok=True)
    ANN_SUBDIR_MMSEG.mkdir(parents=True, exist_ok=True)

    # Process full dataset
    print("\nProcessing full dataset...")
    
    # Process training data
    process_dataset_split("train_data.json", "train")

    # Process validation data
    process_dataset_split("val_data.json", "val")

    print("\n------------------------------------------")
    print("Full dataset processing complete!")
    print(f"Output MMSegmentation data prepared in: {OUTPUT_MMSEG_DIR}")
    print(f"Number of classes defined (excluding background): {len(CLASS_MAPPING)}")
    print(f"Class names: {CLASS_NAMES}")
    print("Remember to update your MMSegmentation config (num_classes, dataset paths, etc.).") 