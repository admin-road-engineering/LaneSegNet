from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import io
import time
import cv2
import numpy as np
from PIL import Image
import base64

# Updated imports for the new structure
from .inference import run_inference, format_results
from .model_loader import load_model
from .schemas import RoadInfrastructureResponse, ErrorResponse, GeographicBounds, InfrastructureElement, AnalysisSummary, ImageMetadata, InfrastructureType, CoordinateAnalysisRequest, ImageAnalysisRequest
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

# Enhanced visualization functions
def get_class_colors():
    """Define colors for different lane marking classes (BGR format for OpenCV)."""
    return {
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

def visualize_lane_markings(image_np: np.ndarray, lane_markings: list, show_labels: bool = True, show_confidence: bool = False) -> np.ndarray:
    """Enhanced lane marking visualization with options.
    
    Args:
        image_np: Original image as NumPy array (BGR format).
        lane_markings: List of lane marking dictionaries from format_results.
        show_labels: Whether to show class labels.
        show_confidence: Whether to show confidence scores.
        
    Returns:
        Image as NumPy array with lane markings drawn.
    """
    class_colors = get_class_colors()
    vis_image = image_np.copy()
    
    # Add semi-transparent overlay for better visibility
    overlay = vis_image.copy()
    
    for marking in lane_markings:
        points = marking.get('points', [])
        class_name = marking.get('class', 'background')
        confidence = marking.get('confidence', 1.0)
        
        if not points or not isinstance(points, list) or len(points) < 2:
            continue
        
        try:
            cv2_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        except ValueError as e:
            logger.warning(f"Could not convert points for visualization: {points} - {e}")
            continue
        
        # Get color for this class
        color = class_colors.get(class_name, (128, 128, 128))
        
        # Adjust thickness based on line type and confidence
        base_thickness = 4 if 'solid' in class_name else 3
        thickness = max(2, int(base_thickness * confidence))
        
        # Draw the lane marking
        cv2.polylines(overlay, [cv2_points], isClosed=False, color=color, thickness=thickness)
        
        # Add labels if requested
        if show_labels and len(points) > 0:
            label_text = class_name
            if show_confidence:
                label_text += f" ({confidence:.2f})"
            
            label_pos = (int(points[0][0]), int(points[0][1]) - 10)
            
            # Add background for text readability
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, 
                         (label_pos[0] - 2, label_pos[1] - text_size[1] - 2),
                         (label_pos[0] + text_size[0] + 2, label_pos[1] + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(overlay, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
    
    return vis_image

def create_side_by_side_visualization(original_image: np.ndarray, lane_markings: list) -> np.ndarray:
    """Create side-by-side comparison of original and annotated images.
    
    Args:
        original_image: Original image (BGR format).
        lane_markings: List of detected lane markings.
        
    Returns:
        Side-by-side comparison image.
    """
    # Create annotated version
    annotated_image = visualize_lane_markings(original_image, lane_markings)
    
    # Ensure both images have the same height
    height = max(original_image.shape[0], annotated_image.shape[0])
    width = original_image.shape[1] + annotated_image.shape[1]
    
    # Create combined image
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place original image on the left
    combined[:original_image.shape[0], :original_image.shape[1]] = original_image
    
    # Place annotated image on the right
    combined[:annotated_image.shape[0], original_image.shape[1]:] = annotated_image
    
    # Add dividing line
    cv2.line(combined, (original_image.shape[1], 0), (original_image.shape[1], height), (255, 255, 255), 2)
    
    # Add labels
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Lane Detection", (original_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined


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
    start_time = time.time()
    if not model:
        logger.warning("Infrastructure analysis endpoint called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        # Validate coordinate bounds
        if coordinates.north <= coordinates.south or coordinates.east <= coordinates.west:
            raise HTTPException(status_code=400, detail="Invalid coordinate bounds: north must be > south, east must be > west")
        
        # Calculate coordinate area for validation
        lat_extent = coordinates.north - coordinates.south
        lon_extent = coordinates.east - coordinates.west
        approx_area_km2 = lat_extent * lon_extent * 111 * 111  # Rough km² calculation
        
        if approx_area_km2 > 100:  # Limit to 100 km² for now
            raise HTTPException(status_code=400, detail=f"Analysis area too large: {approx_area_km2:.2f} km². Maximum allowed: 100 km²")
        
        logger.info(f"Starting infrastructure analysis for coordinates: {coordinates}")
        logger.info(f"Analysis type: {analysis_type}, Resolution: {resolution} m/px, Model: {model_type}")
        
        # --- Imagery Acquisition Pipeline ---
        logger.info("Acquiring aerial imagery from coordinate bounds...")
        try:
            imagery_result = await imagery_manager.acquire_imagery(
                bounds=coordinates,
                resolution_mpp=resolution,
                preferred_provider="local"  # Use local aerial imagery for testing (7,819 images available)
            )
            
            if not imagery_result or "image" not in imagery_result:
                raise RuntimeError("Failed to acquire aerial imagery for the specified coordinates")
            
            img_np_rgb = imagery_result["image"]  # Should be RGB numpy array
            image_metadata = imagery_result.get("metadata", {})
            original_shape = (img_np_rgb.shape[1], img_np_rgb.shape[0])  # (width, height)
            
            logger.info(f"Acquired imagery: {original_shape[0]}x{original_shape[1]} pixels from {image_metadata.get('source', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Imagery acquisition failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to acquire aerial imagery: {e}")

        # --- Infrastructure Detection Pipeline ---
        logger.info("Running infrastructure detection model...")
        try:
            segmentation_map = run_inference(model, img_np_rgb)
            if segmentation_map is None:
                raise RuntimeError("Infrastructure detection inference failed.")
        except RuntimeError as e:
            logger.error(f"Infrastructure inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Infrastructure detection failed: {e}")

        # --- Coordinate Transformation Setup ---
        transformer = CoordinateTransformer(
            bounds=coordinates,
            image_width=original_shape[0],
            image_height=original_shape[1],
            resolution_mpp=resolution
        )
        analyzer = GeographicAnalyzer(transformer)

        # --- Extract and Format Infrastructure Results ---
        try:
            # Get basic detection results
            detection_results = format_results(segmentation_map, original_shape)
            if "error" in detection_results and detection_results["error"]:
                raise ValueError(detection_results["error"])
            
            # Transform to comprehensive infrastructure format
            infrastructure_elements = []
            lane_markings = detection_results.get("lane_markings", [])
            
            for idx, marking in enumerate(lane_markings):
                # Convert to geographic coordinates
                pixel_points = marking.get("points", [])
                geographic_points = transformer.transform_polyline_to_geographic(pixel_points)
                
                # Calculate measurements
                length_pixels = len(pixel_points) * 2 if pixel_points else 0  # Rough estimate
                length_meters = transformer.pixel_distance_to_meters(length_pixels)
                area_pixels = length_pixels * 2  # Rough line area estimate
                area_sqm = transformer.calculate_area_sqm(area_pixels)
                
                # Create infrastructure element
                element = InfrastructureElement(
                    **{"class": marking.get("class", "unknown")},  # Use the 'class' field properly
                    class_id=marking.get("class_id", 0),
                    infrastructure_type=InfrastructureType.LANE_MARKING,
                    points=pixel_points,
                    geographic_points=geographic_points,
                    confidence=marking.get("confidence", 1.0),
                    area_pixels=area_pixels,
                    area_sqm=area_sqm,
                    length_pixels=length_pixels,
                    length_meters=length_meters
                )
                infrastructure_elements.append(element)
            
            # Create analysis summary
            class_counts = detection_results.get("class_summary", {})
            analysis_summary = AnalysisSummary(
                total_elements=len(infrastructure_elements),
                elements_by_type={"lane_marking": len(infrastructure_elements)},
                elements_by_class=class_counts,
                total_lane_length_m=sum(e.length_meters or 0 for e in infrastructure_elements)
            )
            
            # Create image metadata
            img_metadata = ImageMetadata(
                width=original_shape[0],
                height=original_shape[1],
                resolution_mpp=resolution,
                bounds=coordinates,
                source=image_metadata.get("source", "aerial_imagery")
            )
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            raise HTTPException(status_code=500, detail=f"Result formatting failed: {e}")

        # --- Response Handling ---
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        logger.info(f"Infrastructure analysis completed in {processing_time_ms:.2f} ms.")
        logger.info(f"Detected {len(infrastructure_elements)} infrastructure elements")

        if not visualize:
            # Return comprehensive infrastructure analysis response
            response = RoadInfrastructureResponse(
                infrastructure_elements=infrastructure_elements,
                analysis_summary=analysis_summary,
                image_metadata=img_metadata,
                processing_time_ms=processing_time_ms,
                model_type=model_type,
                analysis_type=analysis_type,
                confidence_threshold=0.5,
                coverage_percentage=100.0
            )
            return response
        else:
            # --- Visualization ---
            logger.info("Generating infrastructure visualization...")
            try:
                img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
                vis_image = create_side_by_side_visualization(img_np_bgr, lane_markings)
                
                # Encode visualization image
                is_success, buffer = cv2.imencode(".jpg", vis_image)
                if not is_success:
                    raise ValueError("Failed to encode visualization image.")
                
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
            except Exception as e:
                logger.error(f"Visualization failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly to maintain status code and detail
        raise http_exc
    except Exception as e:
        logger.exception(f"An unexpected error occurred during infrastructure analysis") 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/analyze_image", 
            response_model=RoadInfrastructureResponse, 
            responses={
                400: {"model": ErrorResponse, "description": "Invalid input data"},
                500: {"model": ErrorResponse, "description": "Internal server error"},
                503: {"model": ErrorResponse, "description": "Model not loaded"}
            },
            summary="Analyze Road Infrastructure from Uploaded Image",
            description="Analyzes road infrastructure from an uploaded aerial image with optional coordinate geo-referencing",
            tags=["Road Infrastructure Analysis"])
async def analyze_image(
    image: UploadFile = File(..., description="Aerial image file (JPG, PNG, TIFF)"),
    analysis_type: str = "comprehensive",
    resolution: float = 0.1,
    model_type: str = "swin",
    coordinates_north: Optional[float] = None,
    coordinates_south: Optional[float] = None, 
    coordinates_east: Optional[float] = None,
    coordinates_west: Optional[float] = None,
    image_source: Optional[str] = None,
    visualize: bool = False
):
    """
    Analyzes road infrastructure from an uploaded aerial image.
    
    **Supported formats:** JPG, PNG, TIFF
    **Optional coordinates:** Provide all 4 coordinate parameters for geo-referencing
    **Output:** Same detailed infrastructure analysis as coordinate-based endpoint
    """
    start_time = time.time()
    if not model:
        logger.warning("Image analysis endpoint called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        # Validate image file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        file_extension = '.' + image.filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")
        
        logger.info(f"Starting image analysis for uploaded file: {image.filename}")
        logger.info(f"Analysis type: {analysis_type}, Resolution: {resolution} m/px, Model: {model_type}")
        
        # --- Image Processing Pipeline ---
        try:
            # Read uploaded image
            image_bytes = await image.read()
            
            # Convert to PIL Image then to numpy array
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array (RGB format)
            img_np_rgb = np.array(pil_image)
            original_shape = (img_np_rgb.shape[1], img_np_rgb.shape[0])  # (width, height)
            
            logger.info(f"Processed uploaded image: {original_shape[0]}x{original_shape[1]} pixels")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process uploaded image: {e}")

        # --- Infrastructure Detection Pipeline ---
        logger.info("Running infrastructure detection model...")
        try:
            segmentation_map = run_inference(model, img_np_rgb)
            if segmentation_map is None:
                raise RuntimeError("Infrastructure detection inference failed.")
        except RuntimeError as e:
            logger.error(f"Infrastructure inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Infrastructure detection failed: {e}")

        # --- Optional Coordinate Transformation Setup ---
        transformer = None
        analyzer = None
        coordinates = None
        
        # Check if all coordinate parameters are provided for geo-referencing
        if all(coord is not None for coord in [coordinates_north, coordinates_south, coordinates_east, coordinates_west]):
            # Validate coordinates
            if coordinates_north <= coordinates_south or coordinates_east <= coordinates_west:
                raise HTTPException(status_code=400, detail="Invalid coordinates: north must be > south, east must be > west")
            
            coordinates = GeographicBounds(
                north=coordinates_north,
                south=coordinates_south, 
                east=coordinates_east,
                west=coordinates_west
            )
            
            transformer = CoordinateTransformer(
                bounds=coordinates,
                image_width=original_shape[0],
                image_height=original_shape[1],
                resolution_mpp=resolution
            )
            analyzer = GeographicAnalyzer(transformer)
            logger.info(f"Geo-referencing enabled with coordinates: {coordinates}")

        # --- Extract and Format Infrastructure Results ---
        try:
            # Get basic detection results
            detection_results = format_results(segmentation_map, original_shape)
            if "error" in detection_results and detection_results["error"]:
                raise ValueError(detection_results["error"])
            
            # Transform to comprehensive infrastructure format
            infrastructure_elements = []
            lane_markings = detection_results.get("lane_markings", [])
            
            for idx, marking in enumerate(lane_markings):
                # Convert to geographic coordinates if transformer available
                pixel_points = marking.get("points", [])
                geographic_points = None
                
                if transformer:
                    geographic_points = transformer.transform_polyline_to_geographic(pixel_points)
                
                # Calculate measurements
                length_pixels = len(pixel_points) * 2 if pixel_points else 0  # Rough estimate
                area_pixels = length_pixels * 2  # Rough line area estimate
                
                # Calculate real-world measurements if transformer available
                length_meters = None
                area_sqm = None
                if transformer:
                    length_meters = transformer.pixel_distance_to_meters(length_pixels)
                    area_sqm = transformer.calculate_area_sqm(area_pixels)
                
                # Create infrastructure element
                element = InfrastructureElement(
                    **{"class": marking.get("class", "unknown")},
                    class_id=marking.get("class_id", 0),
                    infrastructure_type=InfrastructureType.LANE_MARKING,
                    points=pixel_points,
                    geographic_points=geographic_points,
                    confidence=marking.get("confidence", 1.0),
                    area_pixels=area_pixels,
                    area_sqm=area_sqm,
                    length_pixels=length_pixels,
                    length_meters=length_meters
                )
                infrastructure_elements.append(element)
            
            # Create analysis summary
            class_counts = detection_results.get("class_summary", {})
            analysis_summary = AnalysisSummary(
                total_elements=len(infrastructure_elements),
                elements_by_type={"lane_marking": len(infrastructure_elements)},
                elements_by_class=class_counts,
                total_lane_length_m=sum(e.length_meters or 0 for e in infrastructure_elements)
            )
            
            # Create image metadata
            img_metadata = ImageMetadata(
                width=original_shape[0],
                height=original_shape[1],
                resolution_mpp=resolution,
                bounds=coordinates,  # Will be None if no coordinates provided
                source=image_source or f"uploaded_{image.filename}"
            )
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            raise HTTPException(status_code=500, detail=f"Result formatting failed: {e}")

        # --- Response Handling ---
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        logger.info(f"Image analysis completed in {processing_time_ms:.2f} ms.")
        logger.info(f"Detected {len(infrastructure_elements)} infrastructure elements")

        if not visualize:
            # Return comprehensive infrastructure analysis response
            response = RoadInfrastructureResponse(
                infrastructure_elements=infrastructure_elements,
                analysis_summary=analysis_summary,
                image_metadata=img_metadata,
                processing_time_ms=processing_time_ms,
                model_type=model_type,
                analysis_type=analysis_type,
                confidence_threshold=0.5,
                coverage_percentage=100.0
            )
            return response
        else:
            # --- Visualization ---
            logger.info("Generating infrastructure visualization...")
            try:
                img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
                vis_image = create_side_by_side_visualization(img_np_bgr, lane_markings)
                
                # Encode visualization image
                is_success, buffer = cv2.imencode(".jpg", vis_image)
                if not is_success:
                    raise ValueError("Failed to encode visualization image.")
                
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
            except Exception as e:
                logger.error(f"Visualization failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly to maintain status code and detail
        raise http_exc
    except Exception as e:
        logger.exception(f"An unexpected error occurred during image analysis") 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/visualize_infrastructure",
         response_class=StreamingResponse,
         summary="Visualize Infrastructure Detection",
         description="Generate visualization of detected infrastructure with original and annotated images side-by-side",
         tags=["Visualization"])
async def visualize_infrastructure(
    coordinates: GeographicBounds,
    resolution: float = 0.1,
    viz_type: str = "side_by_side",  # "side_by_side", "overlay", "original", "annotated"
    show_labels: bool = True,
    show_confidence: bool = False
):
    """Generate visualization of infrastructure detection results.
    
    **Visualization types:**
    - side_by_side: Original and annotated images side by side
    - overlay: Semi-transparent overlay on original
    - original: Original image only
    - annotated: Annotated image only
    """
    start_time = time.time()
    
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")
    
    try:
        # Validate coordinates
        if coordinates.north <= coordinates.south or coordinates.east <= coordinates.west:
            raise HTTPException(status_code=400, detail="Invalid coordinate bounds")
        
        logger.info(f"Generating visualization for coordinates: {coordinates}")
        
        # Acquire imagery
        imagery_result = await imagery_manager.acquire_imagery(
            bounds=coordinates,
            resolution_mpp=resolution,
            preferred_provider="local"
        )
        
        if not imagery_result or "image" not in imagery_result:
            raise HTTPException(status_code=500, detail="Failed to acquire aerial imagery")
        
        img_np_rgb = imagery_result["image"]
        original_shape = (img_np_rgb.shape[1], img_np_rgb.shape[0])
        
        # Run inference
        segmentation_map = run_inference(model, img_np_rgb)
        if segmentation_map is None:
            raise HTTPException(status_code=500, detail="Infrastructure detection failed")
        
        # Get detection results
        detection_results = format_results(segmentation_map, original_shape)
        lane_markings = detection_results.get("lane_markings", [])
        
        # Convert to BGR for OpenCV
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Generate visualization based on type
        if viz_type == "side_by_side":
            vis_image = create_side_by_side_visualization(img_np_bgr, lane_markings)
        elif viz_type == "overlay":
            vis_image = visualize_lane_markings(img_np_bgr, lane_markings, show_labels, show_confidence)
        elif viz_type == "original":
            vis_image = img_np_bgr
        elif viz_type == "annotated":
            # Create pure annotation on black background
            vis_image = np.zeros_like(img_np_bgr)
            vis_image = visualize_lane_markings(vis_image, lane_markings, show_labels, show_confidence)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid visualization type: {viz_type}")
        
        # Add metadata overlay
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        metadata_text = [
            f"Processing: {processing_time:.1f}ms",
            f"Elements: {len(lane_markings)}",
            f"Classes: {len(set(m.get('class', 'unknown') for m in lane_markings))}",
            f"Resolution: {resolution}m/px"
        ]
        
        for i, text in enumerate(metadata_text):
            y_pos = vis_image.shape[0] - (len(metadata_text) - i) * 25 - 10
            cv2.rectangle(vis_image, (5, y_pos - 15), (200, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode and return
        is_success, buffer = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode visualization")
        
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Visualization generation failed")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {e}")

@app.get("/visualizer",
         response_class=HTMLResponse,
         summary="Lane Detection Visualizer Interface",
         description="Interactive web interface for visualizing lane detection results",
         tags=["Visualization"])
async def visualizer_interface():
    """Serve interactive visualization web interface."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>LaneSegNet Visualizer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .control-group { display: flex; flex-direction: column; }
        label { font-weight: bold; margin-bottom: 5px; color: #333; }
        input, select, button { padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button { background: #007bff; color: white; cursor: pointer; font-weight: bold; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .visualization { text-align: center; margin-top: 20px; }
        .result-image { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .loading { display: none; text-align: center; padding: 20px; }
        .error { color: red; text-align: center; padding: 10px; background: #ffe6e6; border-radius: 4px; margin: 10px 0; }
        .success { color: green; text-align: center; padding: 10px; background: #e6ffe6; border-radius: 4px; margin: 10px 0; }
        .presets { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
        .preset-btn { padding: 5px 10px; background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; }
        .preset-btn:hover { background: #1e7e34; }
        .legend { margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 4px; }
        .legend-item { display: inline-block; margin: 5px 10px; }
        .color-box { display: inline-block; width: 20px; height: 15px; margin-right: 5px; vertical-align: middle; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛣️ LaneSegNet Visualizer</h1>
            <p>Interactive lane marking detection and visualization system</p>
        </div>
        
        <div class="presets">
            <button class="preset-btn" onclick="setPreset('brisbane')">Brisbane CBD</button>
            <button class="preset-btn" onclick="setPreset('sydney')">Sydney Harbour Bridge</button>
            <button class="preset-btn" onclick="setPreset('melbourne')">Melbourne CBD</button>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="north">North (Latitude):</label>
                <input type="number" id="north" step="0.000001" value="-27.4698" required>
            </div>
            <div class="control-group">
                <label for="south">South (Latitude):</label>
                <input type="number" id="south" step="0.000001" value="-27.4705" required>
            </div>
            <div class="control-group">
                <label for="east">East (Longitude):</label>
                <input type="number" id="east" step="0.000001" value="153.0258" required>
            </div>
            <div class="control-group">
                <label for="west">West (Longitude):</label>
                <input type="number" id="west" step="0.000001" value="153.0251" required>
            </div>
            <div class="control-group">
                <label for="resolution">Resolution (m/px):</label>
                <input type="number" id="resolution" step="0.01" value="0.1" min="0.05" max="2.0">
            </div>
            <div class="control-group">
                <label for="viz_type">Visualization Type:</label>
                <select id="viz_type">
                    <option value="side_by_side">Side by Side</option>
                    <option value="overlay">Overlay</option>
                    <option value="annotated">Annotated Only</option>
                    <option value="original">Original Only</option>
                </select>
            </div>
            <div class="control-group">
                <label>
                    <input type="checkbox" id="show_labels" checked> Show Labels
                </label>
                <label>
                    <input type="checkbox" id="show_confidence"> Show Confidence
                </label>
            </div>
            <div class="control-group">
                <button onclick="generateVisualization()">Generate Visualization</button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <p>⏳ Processing aerial imagery and detecting lane markings...</p>
        </div>
        
        <div id="message"></div>
        
        <div class="visualization" id="visualization"></div>
        
        <div class="legend">
            <h3>Lane Marking Classes</h3>
            <div class="legend-item"><span class="color-box" style="background: white; border: 2px solid black;"></span>Single White Solid</div>
            <div class="legend-item"><span class="color-box" style="background: #c8c8c8;"></span>Single White Dashed</div>
            <div class="legend-item"><span class="color-box" style="background: yellow;"></span>Single Yellow Solid</div>
            <div class="legend-item"><span class="color-box" style="background: #c8c800;"></span>Single Yellow Dashed</div>
            <div class="legend-item"><span class="color-box" style="background: lime;"></span>Road Edge</div>
            <div class="legend-item"><span class="color-box" style="background: blue;"></span>Crosswalk</div>
            <div class="legend-item"><span class="color-box" style="background: red;"></span>Center Line</div>
            <div class="legend-item"><span class="color-box" style="background: magenta;"></span>Lane Divider</div>
        </div>
    </div>
    
    <script>
        function setPreset(location) {
            const presets = {
                'brisbane': { north: -27.4698, south: -27.4705, east: 153.0258, west: 153.0251 },
                'sydney': { north: -33.8518, south: -33.8525, east: 151.2108, west: 151.2101 },
                'melbourne': { north: -37.8136, south: -37.8143, east: 144.9631, west: 144.9624 }
            };
            
            const coords = presets[location];
            if (coords) {
                document.getElementById('north').value = coords.north;
                document.getElementById('south').value = coords.south;
                document.getElementById('east').value = coords.east;
                document.getElementById('west').value = coords.west;
            }
        }
        
        async function generateVisualization() {
            const button = document.querySelector('button');
            const loading = document.getElementById('loading');
            const message = document.getElementById('message');
            const visualization = document.getElementById('visualization');
            
            // Get form values
            const data = {
                north: parseFloat(document.getElementById('north').value),
                south: parseFloat(document.getElementById('south').value),
                east: parseFloat(document.getElementById('east').value),
                west: parseFloat(document.getElementById('west').value),
                resolution: parseFloat(document.getElementById('resolution').value),
                viz_type: document.getElementById('viz_type').value,
                show_labels: document.getElementById('show_labels').checked,
                show_confidence: document.getElementById('show_confidence').checked
            };
            
            // Validate inputs
            if (data.north <= data.south || data.east <= data.west) {
                message.innerHTML = '<div class="error">Invalid coordinates: North must be > South, East must be > West</div>';
                return;
            }
            
            // Show loading state
            button.disabled = true;
            loading.style.display = 'block';
            message.innerHTML = '';
            visualization.innerHTML = '';
            
            try {
                const response = await fetch('/visualize_infrastructure?' + new URLSearchParams(data), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        north: data.north,
                        south: data.south,
                        east: data.east,
                        west: data.west
                    })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    
                    visualization.innerHTML = `
                        <img src="${imageUrl}" alt="Lane Detection Visualization" class="result-image">
                        <p>✅ Visualization generated successfully!</p>
                    `;
                    
                    message.innerHTML = '<div class="success">Lane detection completed successfully!</div>';
                } else {
                    const errorText = await response.text();
                    message.innerHTML = `<div class="error">Error: ${errorText}</div>`;
                }
            } catch (error) {
                message.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
            } finally {
                button.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter key to trigger visualization
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generateVisualization();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

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