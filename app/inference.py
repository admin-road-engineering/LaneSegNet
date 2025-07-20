import logging
import numpy as np
import torch
import cv2
from typing import Dict
from mmseg.apis import inference_model

from .model_loader import DEVICE, get_lane_classes, get_num_classes

logger = logging.getLogger(__name__)

# Lane marking detection configuration
LANE_CLASSES = get_lane_classes()
NUM_CLASSES = get_num_classes()

# ADE20K to Lane Marking class mapping
# ADE20K has 150 classes, we need to map relevant classes to our 12 lane marking classes
ADE20K_TO_LANE_MAPPING = {
    6: 1,   # road -> single_white_solid (as default lane marking)
    11: 2,  # sidewalk -> single_white_dashed (approximate mapping)
    # Additional mappings can be added based on ADE20K class analysis
    # For now, we'll primarily use class 6 (road) and post-process to detect lane markings
}

# Post-processing parameters for lane marking extraction
MIN_CONTOUR_AREA = 50  # Minimum area for lane marking segments
CONTOUR_APPROX_EPSILON = 0.002  # Approximation epsilon for polyline fitting
MIN_LANE_LENGTH = 20  # Minimum length for a valid lane marking (pixels)

# Enhanced computer vision parameters for lane marking detection
LANE_DETECTION_PARAMS = {
    'gaussian_blur_kernel': (5, 5),
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    'hough_rho': 1,
    'hough_theta': np.pi/180,
    'hough_threshold': 50,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 10,
    'lane_width_pixels': [2, 3, 4, 5, 6],  # Expected lane marking widths
    'white_lower_hsv': np.array([0, 0, 200]),
    'white_upper_hsv': np.array([180, 30, 255]),
    'yellow_lower_hsv': np.array([20, 100, 100]),
    'yellow_upper_hsv': np.array([30, 255, 255])
}

def detect_lane_markings_cv(image_rgb: np.ndarray, road_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Enhanced computer vision lane marking detection within road segments.
    
    Args:
        image_rgb: Original RGB image
        road_mask: Binary mask of road areas from segmentation
        
    Returns:
        Dictionary of lane marking masks by type
    """
    # Convert to different color spaces for analysis
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply road mask to focus only on road areas
    road_only_gray = cv2.bitwise_and(image_gray, image_gray, mask=road_mask.astype(np.uint8))
    road_only_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=np.stack([road_mask]*3, axis=2).astype(np.uint8))
    
    lane_masks = {}
    
    # 1. Detect white lane markings using HSV thresholding
    white_mask = cv2.inRange(road_only_hsv, LANE_DETECTION_PARAMS['white_lower_hsv'], LANE_DETECTION_PARAMS['white_upper_hsv'])
    white_mask = cv2.bitwise_and(white_mask, road_mask.astype(np.uint8))
    
    # 2. Detect yellow lane markings using HSV thresholding  
    yellow_mask = cv2.inRange(road_only_hsv, LANE_DETECTION_PARAMS['yellow_lower_hsv'], LANE_DETECTION_PARAMS['yellow_upper_hsv'])
    yellow_mask = cv2.bitwise_and(yellow_mask, road_mask.astype(np.uint8))
    
    # 3. Edge detection for lane boundaries
    blurred = cv2.GaussianBlur(road_only_gray, LANE_DETECTION_PARAMS['gaussian_blur_kernel'], 0)
    edges = cv2.Canny(blurred, LANE_DETECTION_PARAMS['canny_low_threshold'], LANE_DETECTION_PARAMS['canny_high_threshold'])
    edges = cv2.bitwise_and(edges, road_mask.astype(np.uint8))
    
    # 4. Hough line detection for structured lane markings
    lines = cv2.HoughLinesP(
        edges,
        LANE_DETECTION_PARAMS['hough_rho'],
        LANE_DETECTION_PARAMS['hough_theta'],
        LANE_DETECTION_PARAMS['hough_threshold'],
        minLineLength=LANE_DETECTION_PARAMS['hough_min_line_length'],
        maxLineGap=LANE_DETECTION_PARAMS['hough_max_line_gap']
    )
    
    # Create line mask from Hough lines
    line_mask = np.zeros_like(road_mask, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # 5. Combine detections intelligently
    # White solid lines: strong white + line structure
    white_solid_mask = cv2.bitwise_and(white_mask, line_mask)
    
    # White dashed lines: moderate white without continuous line structure
    white_dashed_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(line_mask))
    
    # Yellow solid lines: strong yellow + line structure
    yellow_solid_mask = cv2.bitwise_and(yellow_mask, line_mask)
    
    # Yellow dashed lines: moderate yellow without continuous line structure
    yellow_dashed_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(line_mask))
    
    # Road edges: edge detection at road boundaries
    road_edge_mask = cv2.bitwise_and(edges, cv2.bitwise_not(cv2.bitwise_or(white_mask, yellow_mask)))
    
    # Clean up masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    lane_masks = {
        'single_white_solid': cv2.morphologyEx(white_solid_mask, cv2.MORPH_CLOSE, kernel),
        'single_white_dashed': cv2.morphologyEx(white_dashed_mask, cv2.MORPH_OPEN, kernel),
        'single_yellow_solid': cv2.morphologyEx(yellow_solid_mask, cv2.MORPH_CLOSE, kernel),
        'single_yellow_dashed': cv2.morphologyEx(yellow_dashed_mask, cv2.MORPH_OPEN, kernel),
        'road_edge': cv2.morphologyEx(road_edge_mask, cv2.MORPH_CLOSE, kernel),
        'center_line': cv2.bitwise_or(yellow_solid_mask, yellow_dashed_mask)  # Center lines are typically yellow
    }
    
    return lane_masks

def map_ade20k_to_lane_classes(segmentation_map: np.ndarray, image_rgb: np.ndarray = None) -> np.ndarray:
    """
    Enhanced mapping from ADE20K to lane classes using computer vision techniques.
    
    Args:
        segmentation_map: Original segmentation map with ADE20K class indices
        image_rgb: Original RGB image for enhanced lane detection
        
    Returns:
        Enhanced segmentation map with detailed lane marking classes
    """
    # Start with basic mapping
    mapped_seg = np.zeros_like(segmentation_map)
    
    # Apply basic class mapping
    for ade20k_class, lane_class in ADE20K_TO_LANE_MAPPING.items():
        mapped_seg[segmentation_map == ade20k_class] = lane_class
    
    # Enhanced processing if image is provided
    if image_rgb is not None:
        # Extract road mask (class 6 in ADE20K)
        road_mask = (segmentation_map == 6).astype(np.uint8)
        
        if road_mask.sum() > 0:  # Only process if road areas detected
            try:
                # Use computer vision to detect detailed lane markings
                lane_masks = detect_lane_markings_cv(image_rgb, road_mask)
                
                # Map detected lane markings to class indices
                class_mapping = {
                    'single_white_solid': 1,
                    'single_white_dashed': 2,
                    'single_yellow_solid': 3,
                    'single_yellow_dashed': 4,
                    'road_edge': 7,
                    'center_line': 8
                }
                
                # Apply detected lane markings with priority (later assignments override)
                for lane_type, mask in lane_masks.items():
                    if lane_type in class_mapping and mask.sum() > MIN_CONTOUR_AREA:
                        mapped_seg[mask > 0] = class_mapping[lane_type]
                        
            except Exception as e:
                logger.warning(f"Enhanced lane detection failed, using basic mapping: {e}")
    
    return mapped_seg

# --- Inference --- 
def run_inference(model, image_np: np.ndarray):
    """
    Runs inference for lane marking detection using MMsegmentation model.
    
    Args:
        model: The loaded MMsegmentation model object for lane detection.
        image_np: Input aerial image as NumPy array (RGB format).
        
    Returns:
        A NumPy array representing the lane marking segmentation map,
        where pixel values are predicted lane marking class indices.
        Returns None on failure.
    """
    logger.info(f"Running lane marking detection inference on aerial image with shape {image_np.shape}...")
    
    try:
        # Run inference - MMsegmentation handles preprocessing internally
        result = inference_model(model, image_np)

        # Extract segmentation map from result
        data_sample = None
        if isinstance(result, list) and len(result) > 0:
            data_sample = result[0]
        elif hasattr(result, 'pred_sem_seg'):
            data_sample = result
        
        if data_sample and hasattr(data_sample, 'pred_sem_seg') and hasattr(data_sample.pred_sem_seg, 'data'):
            seg_tensor = data_sample.pred_sem_seg.data
            raw_segmentation_map = seg_tensor.squeeze().cpu().numpy()
            
            # Check if we need to map from ADE20K classes to lane marking classes
            max_class = raw_segmentation_map.max()
            if max_class > NUM_CLASSES:
                # This is likely ADE20K output, map to lane classes with enhanced CV processing
                logger.info(f"Mapping ADE20K output (max class: {max_class}) to lane marking classes with CV enhancement")
                segmentation_map = map_ade20k_to_lane_classes(raw_segmentation_map, image_np)
            else:
                # This is already in lane marking class format
                segmentation_map = raw_segmentation_map
            
            # Log statistics about detected lane markings
            unique_classes = np.unique(segmentation_map)
            detected_classes = [LANE_CLASSES[i] for i in unique_classes if i < len(LANE_CLASSES)]
            logger.info(f"Lane marking detection successful. Segmentation map shape: {segmentation_map.shape}")
            logger.info(f"Detected lane marking classes: {detected_classes}")
            
            return segmentation_map
        else:
            logger.error("Unexpected inference result format - missing segmentation data.")
            logger.error(f"Result type: {type(result)}")
            return None

    except Exception as e:
        logger.exception(f"Error during lane marking detection inference: {e}")
        return None

# --- Postprocessing --- 
def format_results(segmentation_map: np.ndarray, original_shape: tuple):
    """
    Processes the segmentation map to extract lane marking segments by class.
    
    Args:
        segmentation_map: NumPy array (H, W) with lane marking class indices.
        original_shape: Tuple (width, height) of the original input image.
        
    Returns:
        Dictionary containing lane marking segments grouped by class.
    """
    if segmentation_map is None:
        return {"lane_markings": [], "error": "Inference produced no segmentation map"}

    try:
        original_width, original_height = original_shape
        lane_markings = []
        
        # Process each lane marking class (skip background class 0)
        for class_idx in range(1, min(NUM_CLASSES, segmentation_map.max() + 1)):
            class_name = LANE_CLASSES[class_idx] if class_idx < len(LANE_CLASSES) else f"unknown_class_{class_idx}"
            
            # Extract binary mask for this lane marking class
            class_mask = (segmentation_map == class_idx).astype(np.uint8) * 255
            
            if class_mask.sum() == 0:
                continue  # No pixels found for this class
            
            # Resize mask to original image dimensions
            resized_mask = cv2.resize(class_mask, (original_width, original_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours for this lane marking class
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter by area
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                
                # Check if contour represents a line-like structure
                perimeter = cv2.arcLength(contour, True)
                if perimeter < MIN_LANE_LENGTH:
                    continue
                
                # Approximate contour to reduce points
                epsilon = CONTOUR_APPROX_EPSILON * perimeter
                approx_poly = cv2.approxPolyDP(contour, epsilon, False)
                
                # Convert to point list
                points = approx_poly.squeeze().tolist()
                
                # Handle edge cases
                if not isinstance(points, list):
                    continue
                    
                # Ensure points are in correct format [[x,y], [x,y], ...]
                if len(points) > 0 and not isinstance(points[0], list):
                    if len(points) == 2:  # Single point [x, y]
                        points = [points]
                    else:
                        continue
                
                if len(points) < 2:  # Need at least 2 points for a line
                    continue
                
                # Determine lane marking properties
                lane_marking = {
                    "class": class_name,
                    "class_id": int(class_idx),
                    "points": points,
                    "confidence": 1.0,  # Could be enhanced with actual confidence scores
                    "area": float(cv2.contourArea(contour)),
                    "length": float(perimeter)
                }
                
                lane_markings.append(lane_marking)
        
        logger.info(f"Extracted {len(lane_markings)} lane marking segments across all classes.")
        
        # Group by class for summary
        class_counts = {}
        for marking in lane_markings:
            class_name = marking["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info(f"Lane marking detection summary: {class_counts}")
        
        return {
            "lane_markings": lane_markings,
            "class_summary": class_counts,
            "total_segments": len(lane_markings)
        }

    except Exception as e:
        logger.exception(f"Error during lane marking result formatting: {e}")
        return {"lane_markings": [], "error": f"Formatting failed: {e}"}

# === Lane Marking Detection Pipeline Complete === 