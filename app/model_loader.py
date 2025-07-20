import logging
import torch
from mmseg.apis import init_model

logger = logging.getLogger(__name__)

# Configuration for lane marking detection models
# Based on the ArXiv paper: https://arxiv.org/html/2410.05717v1
# Supports 12-14 lane marking classes for aerial imagery

# Swin Transformer configuration for semantic segmentation
CONFIG_FILE = 'configs/mmseg/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
CHECKPOINT_FILE = 'weights/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth'

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

NUM_CLASSES = len(LANE_CLASSES)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(use_lanesegnet=False):
    """
    Loads the MMsegmentation model for lane marking detection.
    
    Args:
        use_lanesegnet: If True, loads the specialized LaneSegNet model.
                       If False, loads the Swin Transformer model.
    
    Returns:
        The loaded model object or None if loading fails.
    """
    if use_lanesegnet:
        config_file = LANESEGNET_CONFIG
        checkpoint_file = LANESEGNET_CHECKPOINT
        model_type = "LaneSegNet"
    else:
        config_file = CONFIG_FILE
        checkpoint_file = CHECKPOINT_FILE
        model_type = "Swin Transformer"
    
    logger.info(f"Attempting to load {model_type} model for lane marking detection...")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Checkpoint file: {checkpoint_file}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Expected lane marking classes: {NUM_CLASSES}")

    try:
        # Initialize the segmentation model
        model = init_model(config_file, checkpoint_file, device=device)
        model.eval()  # Set model to evaluation mode
        logger.info(f"{model_type} model loaded successfully for lane marking detection.")
        return model
    except FileNotFoundError as e:
        logger.error(f"Error: Config file '{config_file}' or Checkpoint file '{checkpoint_file}' not found.")
        logger.error(f"Please ensure model weights are downloaded to the weights/ directory.")
        return None
    except Exception as e:
        logger.exception(f"An error occurred during {model_type} model loading: {e}")
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