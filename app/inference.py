import logging
import numpy as np
import torch
import cv2
from mmseg.apis import inference_model

from .model_loader import DEVICE, get_lane_classes, get_num_classes

logger = logging.getLogger(__name__)

# Lane marking detection configuration
LANE_CLASSES = get_lane_classes()
NUM_CLASSES = get_num_classes()

# Post-processing parameters for lane marking extraction
MIN_CONTOUR_AREA = 50  # Minimum area for lane marking segments
CONTOUR_APPROX_EPSILON = 0.002  # Approximation epsilon for polyline fitting
MIN_LANE_LENGTH = 20  # Minimum length for a valid lane marking (pixels)

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
            segmentation_map = seg_tensor.squeeze().cpu().numpy()
            
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