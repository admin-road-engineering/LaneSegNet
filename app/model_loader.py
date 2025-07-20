import logging
import torch
# NEW: Import from mmseg
from mmseg.apis import init_model
import segmentation_models_pytorch as smp
import os

logger = logging.getLogger(__name__)

# Define paths for the new Swin Transformer model
CONFIG_FILE = 'configs/mmseg/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
CHECKPOINT_FILE = 'weights/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth'

# --- Model Configuration ---
# NOTE: Confirm this matches the backbone used for the downloaded 'best.pth'
# The sabadijou repo defaults to 'resnet34'.
ENCODER = 'resnet34' 
ENCODER_WEIGHTS = 'imagenet' # Use ImageNet pretraining for the encoder
MODEL_WEIGHTS_PATH = "weights/best.pth" # Path to the downloaded weights
# CLASSES = ['road'] # Assuming the model outputs a single class mask for roads/lanes
NUM_CLASSES = 6 # IMPORTANT: Match the number of classes the pre-trained model expects (Dubai dataset: Building, Land, Road, Vegetation, Water, Unlabeled)
ACTIVATION = None # Use None for multi-class output (logits), apply sigmoid/softmax later if needed
# --- End Model Configuration ---

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """
    Loads the MMsegmentation model using config and checkpoint.
    Returns:
        The loaded model object or None if loading fails.
    """
    logger.info(f"Attempting to load MMsegmentation model...")
    logger.info(f"Config file: {CONFIG_FILE}")
    logger.info(f"Checkpoint file: {CHECKPOINT_FILE}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    try:
        # Initialize the segmentor
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
        model.eval() # Set model to evaluation mode
        logger.info("MMsegmentation model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Error: Config file '{CONFIG_FILE}' or Checkpoint file '{CHECKPOINT_FILE}' not found.")
        return None
    except Exception as e:
        logger.exception(f"An error occurred during MMsegmentation model loading: {e}")
        return None

# --- Keep the old load_model (renamed) temporarily if needed for comparison ---
# def load_unet_model(): ... (your previous code) ...

def load_model_smp():
    """
    Loads the pre-trained U-Net segmentation model.
    Returns:
        The loaded model object or None if loading fails.
    """
    logger.info(f"Attempting to load segmentation model with {ENCODER} backbone.")
    logger.info(f"Attempting to load weights from: {MODEL_WEIGHTS_PATH}")
    logger.info(f"Using device: {DEVICE}")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logger.error(f"Model weights file not found at expected path: /app/{MODEL_WEIGHTS_PATH}")
        return None

    try:
        # Create U-Net model architecture - matching the pre-trained model's output classes
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=NUM_CLASSES, # Use the number of classes from the checkpoint
            activation=ACTIVATION, # Output raw logits for multi-class
        )

        # Load the downloaded weights
        # Use map_location to ensure compatibility if weights were saved on a different device
        logger.info(f"Loading state dict from {MODEL_WEIGHTS_PATH}...")
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device(DEVICE))
        
        # Check if the checkpoint contains keys like 'state_dict' or is the state_dict directly
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
             checkpoint = state_dict # Keep the original checkpoint if needed
             state_dict = state_dict['state_dict']
             logger.info("Extracted state_dict from checkpoint object.")
        
        # --- Adjust keys: Remove potential "unet." prefix --- 
        # Check if keys start with "unet." and rename if necessary
        if all(key.startswith('unet.') for key in state_dict.keys()):
            logger.info("Removing 'unet.' prefix from state_dict keys.")
            state_dict = {k.replace('unet.', '', 1): v for k, v in state_dict.items()}
        # --- End Key Adjustment ---

        # Optional: Adjust keys if needed (e.g., remove 'module.' prefix if saved with DataParallel)
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        logger.info("Weights loaded successfully into model structure.")

        # Move model to the appropriate device and set to evaluation mode
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model ready on device: {DEVICE}")
        
        return model

    except ImportError as e:
        logger.error(f"ImportError during model loading. Ensure PyTorch ({torch.__version__}) and segmentation-models-pytorch are installed correctly: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found during model loading: {e}")
        return None
    except Exception as e:
        logger.exception("An unexpected error occurred during model loading")
        return None 