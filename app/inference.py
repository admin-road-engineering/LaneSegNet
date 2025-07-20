import logging
from PIL import Image
import io
import numpy as np
import torch
import cv2 # OpenCV for post-processing
from skimage.morphology import skeletonize # For centerline extraction
import segmentation_models_pytorch as smp
from mmseg.apis import inference_model

# Import necessary MMDetection/MMSegmentation/MMDetection3D components here
# e.g., from mmcv.parallel import collate, scatter
# from mmseg.apis import inference_segmentor
# from mmseg.datasets.pipelines import Compose

# Ensure model_loader provides the correct device
from .model_loader import DEVICE, ENCODER # Import device and encoder info

# Assuming the Dubai dataset class indices from sabadijou/dataset/dataset.py
# 0: Unlabeled, 1: Building, 2: Land, 3: Road, 4: Vegetation, 5: Water
ROAD_CLASS_INDEX = 6 # 'road' is index 6 in ADE20K
MIN_CONTOUR_AREA = 100 # Keep or adjust as needed

logger = logging.getLogger(__name__)

# --- Preprocessing --- 
def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the input image bytes for the segmentation model.
    Args:
        image_bytes: Raw bytes of the uploaded image.
    Returns:
        Preprocessed image tensor ready for the model.
    """
    logger.info("Preprocessing image for segmentation model...")
    try:
        # Load image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        original_h, original_w = image_np.shape[:2]
        logger.debug(f"Original image shape: H={original_h}, W={original_w}")

        # --- Padding Logic --- 
        # Calculate required padding to make dimensions divisible by 32
        h, w = image_np.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        # Determine padding amounts for top/bottom/left/right
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if pad_h > 0 or pad_w > 0:
             logger.info(f"Padding image from ({h}, {w}) to ({h+pad_h}, {w+pad_w}) to be divisible by 32.")
             # Pad the image using OpenCV with constant border (black)
             padded_image_np = cv2.copyMakeBorder(image_np, top, bottom, left, right, 
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
             h, w = padded_image_np.shape[:2] # Update dimensions
             logger.debug(f"Padded image shape: H={h}, W={w}")
        else:
             padded_image_np = image_np # No padding needed
        # --- End Padding Logic ---

        # Get preprocessing function from segmentation_models_pytorch
        # Uses normalization parameters corresponding to the ImageNet-pretrained encoder
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained='imagenet')
        # Apply preprocessing to the padded image
        preprocessed_np = preprocessing_fn(padded_image_np)
        
        # Convert to PyTorch tensor -> (H, W, C) to (C, H, W)
        # Ensure the tensor is float32 to match model weights
        image_tensor = torch.from_numpy(preprocessed_np).permute(2, 0, 1).contiguous().float()
        
        # Add batch dimension -> (C, H, W) to (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(DEVICE)
        
        logger.info("Image preprocessing complete.")
        return image_tensor
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}", exc_info=True)
        raise ValueError(f"Image preprocessing failed: {e}")

# --- Inference --- 
def run_inference(model, image_np: np.ndarray):
    """
    Runs inference using the loaded MMsegmentation model.
    Args:
        model: The loaded MMsegmentation model object.
        image_np: Input image as a NumPy array (RGB format).
    Returns:
        A NumPy array representing the segmentation map, where pixel values
        are the predicted class indices. Returns None on failure.
    """
    logger.info(f"Running MMsegmentation inference on image with shape {image_np.shape}...")
    try:
        # MMsegmentation inference_model handles preprocessing based on config
        # It expects an image path or a numpy array.
        # If image_np is RGB, mmseg will handle BGR conversion if its internal pipeline expects BGR.
        # Or, explicitly convert to BGR if testing shows it's necessary:
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        result = inference_model(model, image_np) # Pass RGB numpy array

        data_sample = None
        if isinstance(result, list) and len(result) > 0:
            data_sample = result[0] # Assuming the first element is the SegDataSample
        elif hasattr(result, 'pred_sem_seg'): # Check if result itself is a SegDataSample
            data_sample = result
        
        if data_sample and hasattr(data_sample, 'pred_sem_seg') and hasattr(data_sample.pred_sem_seg, 'data'):
            seg_tensor = data_sample.pred_sem_seg.data
            segmentation_map = seg_tensor.squeeze().cpu().numpy()
            logger.info(f"Inference successful. Segmentation map shape: {segmentation_map.shape}")
            return segmentation_map
        else:
            logger.error(f"Unexpected inference result format or missing pred_sem_seg.data.")
            logger.error(f"Type of result: {type(result)}")
            if isinstance(result, list) and len(result) > 0:
                 logger.error(f"Type of result[0]: {type(result[0])}, Attributes: {dir(result[0]) if result[0] else 'None'}")
            elif data_sample:
                 logger.error(f"Data_sample type: {type(data_sample)}, Attributes: {dir(data_sample)}")
            else:
                 logger.error("Result was not a list and not a recognizable SegDataSample.")

    except Exception as e:
        logger.exception(f"Error during MMsegmentation inference: {e}")
        return None

# --- Postprocessing --- 
def format_results(segmentation_map: np.ndarray, original_shape: tuple):
    """
    Processes the raw segmentation map to extract lane polylines.
    Args:
        segmentation_map: NumPy array (H, W) with class indices.
        original_shape: Tuple (width, height) of the original input image.
    Returns:
        Dictionary containing lane segments.
    """
    if segmentation_map is None:
         return {"lanes": [], "error": "Inference produced no map"}

    try:
        original_height, original_width = original_shape[1], original_shape[0]
        # map_height, map_width = segmentation_map.shape # Unused

        # 1. Extract the binary mask for the road class
        road_mask = (segmentation_map == ROAD_CLASS_INDEX).astype(np.uint8) * 255
        logger.info(f"Extracted binary mask for class {ROAD_CLASS_INDEX} (road). Sum of mask pixels: {road_mask.sum()}")

        if road_mask.sum() == 0:
             logger.warning(f"No pixels found for road class index {ROAD_CLASS_INDEX}.")
             return {"lanes": []}

        # 2. Resize mask to original image dimensions
        resized_mask = cv2.resize(road_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # No thresholding needed here as we are working with class indices directly.
        # processed_mask = resized_mask \
        
        # 3. Skeletonization (optional, but can be good for line-like structures)
        # For skeletonization, ensure the input is a binary image (0 or 255)
        # skeleton = cv2.ximgproc.thinning(resized_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # logger.info(f"Skeleton mask sum: {skeleton.sum()}")
        # mask_for_contours = skeleton
        mask_for_contours = resized_mask # Using resized mask directly for now
        
        # 4. Find Contours
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} raw contours for road class.")

        # 5. Filter and Format Contours
        lane_segments = []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, False)

            points = approx_poly.squeeze().tolist()
            if not isinstance(points, list) or (isinstance(points, list) and len(points) > 0 and not isinstance(points[0], list)):
                # Handle cases where squeeze might result in a single point or a flat list for a very short line
                if isinstance(points, list) and len(points) == 2 and isinstance(points[0], (int, float)): # Single point [x,y]
                     points = [points] # Wrap it as [[x,y]]
                elif not isinstance(points, list): # Squeezed to a non-list (e.g. if contour was a single point itself)
                    continue # or handle as a single point if meaningful
           
            if len(points) < 2: # Need at least two points for a line segment
                 continue

            lane_type = "solid" # Defaulting to solid for now

            lane_segments.append({
                "points": points,
                "type": lane_type
            })

        logger.info(f"Formatted {len(lane_segments)} lane segments after filtering.")
        return {"lanes": lane_segments}

    except Exception as e:
        logger.exception(f"Error during result formatting: {e}")
        return {"lanes": [], "error": f"Formatting failed: {e}"}

# --- Old U-Net/SMP specific functions - to be removed or kept for reference only ---
# def preprocess_image_smp(...):
# def run_inference_smp(...):
# def format_results_smp(...): 