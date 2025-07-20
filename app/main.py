from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import io
import time
import cv2
import numpy as np
from PIL import Image

# Updated imports for the new structure
from .inference import run_inference, format_results
from .model_loader import load_model
from .schemas import LaneDetectionResponse, ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aerial Lane Detection API", version="0.3.0")

# Global variable to hold the segmentation model
model = None
# model_config is removed as it's not needed for this model type

@app.on_event("startup")
def startup_event(): # Can be sync if model loading is sync
    global model
    logger.info("Loading MMsegmentation model (Swin Transformer)...")
    model = load_model() # Call the updated loader
    if model:
        logger.info("MMsegmentation model loaded successfully.")
    else:
        logger.error("Failed to load MMsegmentation model. API will not function correctly.")
        # Consider raising an error to stop startup if model is essential
        # raise RuntimeError("Could not load the segmentation model")

# Helper function for visualization
def visualize_lanes(image_np: np.ndarray, lanes_data: list) -> np.ndarray:
    """Draws detected lanes on the image.
    Args:
        image_np: Original image as a NumPy array (BGR format).
        lanes_data: List of lane segment dictionaries from format_results.
    Returns:
        Image as NumPy array with lanes drawn.
    """
    vis_image = image_np.copy()
    for lane in lanes_data:
        points = lane.get('points', [])
        lane_type = lane.get('type', 'solid') # Default to solid if type is missing
        if not points or not isinstance(points, list) or len(points) < 2:
            continue
        
        # Ensure points are in the correct format for cv2.polylines, e.g., List[List[int]] or List[Tuple[int, int]]
        # The points from approxPolyDP are already List[List[int]] after .tolist()
        # if they are flat List[int] from a single point, format_results should handle it.
        try:
            cv2_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        except ValueError as e:
            logger.warning(f"Could not convert points to NumPy array for visualization: {points} - {e}")
            continue
        
        color = (0, 0, 255) # Red for all detected roads for now (BGR)
        cv2.polylines(vis_image, [cv2_points], isClosed=False, color=color, thickness=2)
    return vis_image


@app.post("/detect_lanes", 
            response_model=LaneDetectionResponse, 
            responses={
                400: {"model": ErrorResponse, "description": "Invalid input image"},
                500: {"model": ErrorResponse, "description": "Internal server error"},
                503: {"model": ErrorResponse, "description": "Model not loaded"}
            },
            summary="Detect Lane Markings in an Aerial Image (Swin Transformer)",
            tags=["Lane Detection"])
async def detect_lanes(image: UploadFile = File(..., description="Aerial image file for lane detection."),
                       visualize: bool = False # Optional query parameter for visualization
                       ):
    """
    Accepts an aerial image file and returns detected lane segments as polylines.

    - **image**: The aerial image file (e.g., JPEG, PNG) to process.
    - **visualize**: Set to `true` to return the image with detected lanes drawn on it instead of JSON data.
    """
    start_time = time.time()
    if not model:
        logger.warning("Detect lanes endpoint called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    if not image.content_type or not image.content_type.startswith('image/'):
        logger.warning(f"Invalid content type received: {image.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image.content_type}'. Please upload an image.")

    try:
        image_bytes = await image.read()
        file_size_kb = len(image_bytes) / 1024
        logger.info(f"Received image: {image.filename}, size: {file_size_kb:.2f} KB, type: {image.content_type}")

        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_pil.verify() # Verify image header and format
            # Re-open after verify to read actual image data
            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_shape = img_pil.size # (width, height)
            img_np_rgb = np.array(img_pil) # Convert PIL RGB to NumPy RGB for inference
            logger.info(f"Image validated: {image.filename}, shape: {original_shape}")
        except Exception as e:
            logger.error(f"Invalid image file provided: {image.filename} - {e}")
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {e}")

        # --- Inference Pipeline --- 
        logger.info(f"Starting MMsegmentation inference pipeline for {image.filename}...")
        
        # 1. Preprocessing - Handled by MMsegmentation's inference_segmentor

        # 2. Inference - Pass NumPy RGB image
        try:
            segmentation_map = run_inference(model, img_np_rgb) # Pass RGB numpy array
            if segmentation_map is None:
                raise RuntimeError("Inference did not return a segmentation map.")
        except RuntimeError as e:
             logger.error(f"Inference failed for {image.filename}: {e}")
             raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

        # 3. Postprocessing & Formatting
        try:
            formatted_response = format_results(segmentation_map, original_shape)
            if "error" in formatted_response and formatted_response["error"]:
                raise ValueError(formatted_response["error"])
        except ValueError as e:
             logger.error(f"Result formatting failed for {image.filename}: {e}")
             raise HTTPException(status_code=500, detail=f"Result formatting failed: {e}")

        # --- Response Handling ---
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        logger.info(f"Inference pipeline completed for {image.filename} in {processing_time_ms:.2f} ms.")

        # Add processing time to JSON response if not visualizing
        if not visualize:
            formatted_response["processing_time_ms"] = processing_time_ms
            # FastAPI will automatically validate this against LaneDetectionResponse
            return formatted_response
        else:
            # --- Visualization --- 
            logger.info(f"Generating visualization for {image.filename}...")
            try:
                img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV drawing
                vis_image = visualize_lanes(img_np_bgr, formatted_response.get("lanes", []))
                
                # Encode image to return
                is_success, buffer = cv2.imencode(".jpg", vis_image)
                if not is_success:
                    raise ValueError("Failed to encode visualization image.")
                
                # Return as image stream
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
            except Exception as e:
                 logger.error(f"Visualization failed for {image.filename}: {e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly to maintain status code and detail
        raise http_exc
    except Exception as e:
        logger.exception(f"An unexpected error occurred during lane detection for {image.filename}") 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/health", 
         summary="Health Check", 
         tags=["Management"],
         responses={200: {"description": "API is running"}})
async def health_check():
    """Provides a basic health check of the API and confirms model status."""
    # Updated check: model should not be None
    return {"status": "ok", "model_loaded": model is not None}

# You can run this directly using `python -m uvicorn app.main:app --reload --port 8010` for development
# The Dockerfile uses `uvicorn app.main:app --host 0.0.0.0 --port 8010` for production
# if __name__ == "__main__":
#     import uvicorn
#     # Note: Running directly like this won't load the model via the startup event
#     # Use 'uvicorn app.main:app --reload --port 8010' instead for development.
#     uvicorn.run(app, host="127.0.0.1", port=8010) 