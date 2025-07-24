import logging
import torch
from mmseg.apis import init_model
import os
import sys

# Import central configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.global_config import NUM_CLASSES as CENTRAL_NUM_CLASSES

logger = logging.getLogger(__name__)

# Configuration for lane marking detection models
# Based on the ArXiv paper: https://arxiv.org/html/2410.05717v1
# Supports 12-14 lane marking classes for aerial imagery

# Phase 2 Enhancement: Prioritize 12-class lane marking configuration
CONFIG_FILE = 'configs/mmseg/swin_base_lane_markings_12class.py'
CHECKPOINT_FILE = 'weights/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth'

# Alternative: Fine-tuned model (if available)
FINE_TUNED_CHECKPOINT = 'weights/best.pth'  # Use if lane-specific training completed

# Alternative: LaneSegNet configuration for specialized lane detection
LANESEGNET_CONFIG = 'configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py'
LANESEGNET_CHECKPOINT = 'weights/lanesegnet_r50_8x1_24e_olv2_subset_A (2).pth'

# Lane marking classes based on SkyScapes/Waterloo datasets
# Update this based on your specific dataset
LANE_CLASSES = [
    'background',
    'single_white_solid',
    'single_white_dashed', 
    'single_yellow_solid',
    'single_yellow_dashed',
    'double_white_solid',
    'double_yellow_solid',
    'road_edge',
    'center_line',
    'lane_divider',
    'crosswalk',
    'stop_line'
]

# Override local with central config (temporarily limit to 3; expand later if needed)
NUM_CLASSES = CENTRAL_NUM_CLASSES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(use_lanesegnet=False, prefer_fine_tuned=True):
    """
    Enhanced model loading with automatic fine-tuned model detection.
    
    Args:
        use_lanesegnet: If True, loads the specialized LaneSegNet model.
                       If False, loads the Swin Transformer model.
        prefer_fine_tuned: If True, tries fine-tuned model first.
    
    Returns:
        The loaded model object or None if loading fails.
    """
    import os
    
    # Phase 2 Enhancement: Prioritize lane-specific fine-tuned models
    checkpoint_candidates = []
    
    if use_lanesegnet:
        config_file = LANESEGNET_CONFIG
        checkpoint_candidates = [LANESEGNET_CHECKPOINT]
        model_type = "LaneSegNet"
    else:
        config_file = CONFIG_FILE
        if prefer_fine_tuned and os.path.exists(FINE_TUNED_CHECKPOINT):
            checkpoint_candidates = [FINE_TUNED_CHECKPOINT, CHECKPOINT_FILE]
            logger.info("Phase 2 Enhancement: Fine-tuned model detected, will try first")
        else:
            checkpoint_candidates = [CHECKPOINT_FILE]
        model_type = "Enhanced Swin Transformer (12-class)"
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Expected lane marking classes: {NUM_CLASSES}")
    
    # Try each checkpoint candidate
    for i, checkpoint_file in enumerate(checkpoint_candidates):
        checkpoint_type = "fine-tuned" if i == 0 and prefer_fine_tuned else "pre-trained"
        logger.info(f"Attempting to load {model_type} model ({checkpoint_type})...")
        logger.info(f"Config file: {config_file}")
        logger.info(f"Checkpoint file: {checkpoint_file}")

        try:
            if not os.path.exists(checkpoint_file):
                logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                continue
                
            # Initialize the segmentation model
            model = init_model(config_file, checkpoint_file, device=device)
            model.eval()  # Set model to evaluation mode
            
            # Validate model output classes
            if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
                model_classes = model.decode_head.num_classes
                logger.info(f"Model configured for {model_classes} classes")
                if model_classes != NUM_CLASSES:
                    logger.warning(f"Class mismatch: Model has {model_classes} classes, expected {NUM_CLASSES}")
            
            logger.info(f"{model_type} model loaded successfully ({checkpoint_type}).")
            logger.info(f"Phase 2 Status: Enhanced CV pipeline active for 80-85% mIoU target")
            return model
            
        except FileNotFoundError as e:
            logger.warning(f"Checkpoint file '{checkpoint_file}' not found, trying next candidate...")
            continue
        except Exception as e:
            logger.warning(f"Failed to load {checkpoint_type} model: {e}")
            continue
    
    # All attempts failed
    logger.error("Failed to load any model checkpoint")
    logger.error("Available candidates tried:")
    for checkpoint in checkpoint_candidates:
        exists = "✓" if os.path.exists(checkpoint) else "✗"
        logger.error(f"  {exists} {checkpoint}")
    return None

def get_lane_classes():
    """
    Returns the list of lane marking classes supported by the model.
    """
    return LANE_CLASSES.copy()

def get_num_classes():
    """
    Returns the number of lane marking classes.
    """
    return NUM_CLASSES 