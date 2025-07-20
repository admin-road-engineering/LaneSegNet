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
from .schemas import RoadInfrastructureResponse, ErrorResponse, GeographicBounds, InfrastructureElement, AnalysisSummary, ImageMetadata
from .imagery_acquisition import imagery_manager
from .coordinate_transform import CoordinateTransformer, GeographicAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aerial Lane Marking Detection API", 
    version="1.0.0",
    description="Deep learning-based lane marking detection from aerial imagery for HD map creation"
)

# Global variable to hold the segmentation model
model = None
# model_config is removed as it's not needed for this model type

@app.on_event("startup")
def startup_event():
    global model
    logger.info("Loading lane marking detection model...")
    model = load_model(use_lanesegnet=False)  # Start with Swin Transformer
    if model:
        logger.info("Lane marking detection model loaded successfully.")
    else:
        logger.error("Failed to load lane marking detection model. API will not function correctly.")
        # For production, uncomment the next line to prevent startup with failed model loading
        # raise RuntimeError("Could not load the lane marking detection model")

# Helper function for visualization
def visualize_lane_markings(image_np: np.ndarray, lane_markings: list) -> np.ndarray:
    """Draws detected lane markings on the image with class-specific colors.
    
    Args:
        image_np: Original image as NumPy array (BGR format).
        lane_markings: List of lane marking dictionaries from format_results.
        
    Returns:
        Image as NumPy array with lane markings drawn.
    """
    # Define colors for different lane marking classes (BGR format)
    class_colors = {
        'single_white_solid': (255, 255, 255),     # White
        'single_white_dashed': (200, 200, 200),   # Light gray
        'single_yellow_solid': (0, 255, 255),     # Yellow
        'single_yellow_dashed': (0, 200, 200),    # Dark yellow
        'double_white_solid': (255, 255, 255),    # White
        'double_yellow_solid': (0, 255, 255),     # Yellow
        'road_edge': (0, 255, 0),                 # Green
        'center_line': (0, 0, 255),               # Red
        'lane_divider': (255, 0, 255),            # Magenta
        'crosswalk': (255, 0, 0),                 # Blue
        'stop_line': (0, 0, 255),                 # Red
        'background': (128, 128, 128)             # Gray
    }
    
    vis_image = image_np.copy()
    
    for marking in lane_markings:
        points = marking.get('points', [])
        class_name = marking.get('class', 'background')
        
        if not points or not isinstance(points, list) or len(points) < 2:
            continue
        
        try:
            cv2_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        except ValueError as e:
            logger.warning(f"Could not convert points for visualization: {points} - {e}")
            continue
        
        # Get color for this class
        color = class_colors.get(class_name, (128, 128, 128))  # Default to gray
        
        # Draw the lane marking
        thickness = 3 if 'solid' in class_name else 2
        cv2.polylines(vis_image, [cv2_points], isClosed=False, color=color, thickness=thickness)
        
        # Add class label near the first point
        if len(points) > 0:
            label_pos = (int(points[0][0]), int(points[0][1]) - 10)
            cv2.putText(vis_image, class_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
    
    return vis_image


@app.post("/analyze_road_infrastructure", 
            response_model=RoadInfrastructureResponse, 
            responses={
                400: {"model": ErrorResponse, "description": "Invalid input data"},
                500: {"model": ErrorResponse, "description": "Internal server error"},
                503: {"model": ErrorResponse, "description": "Model not loaded"}
            },
            summary="Analyze Road Infrastructure from Coordinates",
            description="Analyzes road infrastructure (lane markings, pavements, footpaths, etc.) from geographic coordinates using aerial imagery",
            tags=["Road Infrastructure Analysis"])
async def analyze_road_infrastructure(
    coordinates: GeographicBounds,  # Lat/lon bounding box from frontend
    analysis_type: str = "comprehensive",  # 'lane_markings', 'pavements', 'comprehensive'
    resolution: float = 0.1,  # meters per pixel
    visualize: bool = False,
    model_type: str = "swin"
):
    """
    Analyzes complete road infrastructure from geographic coordinates.
    
    **Supported infrastructure types:**
    - Lane markings: solid lines, dashed lines, crosswalks
    - Road surfaces: asphalt, concrete, gravel, dirt
    - Pedestrian infrastructure: sidewalks, footpaths, bike lanes
    - Road features: edges, curbs, medians, shoulders
    - Utilities: storm drains, utility covers, parking spaces
    
    **Input**: Geographic bounding box coordinates from road-engineering frontend
    **Output**: Detailed infrastructure analysis with geometric data
    """
    """
    Detects and classifies lane markings from aerial imagery.
    
    **Supported lane marking classes:**
    - single_white_solid, single_white_dashed
    - single_yellow_solid, single_yellow_dashed  
    - double_white_solid, double_yellow_solid
    - road_edge, center_line, lane_divider
    - crosswalk, stop_line

    - **image**: The aerial image file (JPEG, PNG, etc.) to analyze
    - **visualize**: Return annotated image instead of JSON data
    - **model_type**: Model to use ('swin' for Swin Transformer, 'lanesegnet' for specialized model)
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

        # --- Lane Marking Detection Pipeline ---
        logger.info(f"Starting lane marking detection for {image.filename}...")
        logger.info(f"Using model type: {model_type}")
        
        # Run inference - MMsegmentation handles preprocessing internally
        try:
            segmentation_map = run_inference(model, img_np_rgb)
            if segmentation_map is None:
                raise RuntimeError("Lane marking detection inference failed.")
        except RuntimeError as e:
            logger.error(f"Lane marking inference failed for {image.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Lane marking detection failed: {e}")

        # Extract and format lane marking results
        try:
            results = format_results(segmentation_map, original_shape)
            if "error" in results and results["error"]:
                raise ValueError(results["error"])
        except ValueError as e:
            logger.error(f"Lane marking result formatting failed for {image.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Result formatting failed: {e}")

        # --- Response Handling ---
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        lane_markings = results.get("lane_markings", [])
        total_segments = results.get("total_segments", 0)
        class_summary = results.get("class_summary", {})
        
        logger.info(f"Lane marking detection completed for {image.filename} in {processing_time_ms:.2f} ms.")
        logger.info(f"Detected {total_segments} lane marking segments: {class_summary}")

        if not visualize:
            # Return JSON response with lane marking data
            response_data = {
                "lane_markings": lane_markings,
                "class_summary": class_summary,
                "total_segments": total_segments,
                "processing_time_ms": processing_time_ms,
                "model_type": model_type,
                "image_shape": {"width": original_shape[0], "height": original_shape[1]}
            }
            return response_data
        else:
            # --- Visualization ---
            logger.info(f"Generating lane marking visualization for {image.filename}...")
            try:
                img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
                vis_image = visualize_lane_markings(img_np_bgr, lane_markings)
                
                # Encode visualization image
                is_success, buffer = cv2.imencode(".jpg", vis_image)
                if not is_success:
                    raise ValueError("Failed to encode visualization image.")
                
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