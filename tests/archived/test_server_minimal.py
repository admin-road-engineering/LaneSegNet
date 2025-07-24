"""
Minimal test server to validate API structure without model dependencies.
This allows us to test the API endpoints and coordinate transformation logic.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Import our schemas
from app.schemas import (
    RoadInfrastructureResponse, ErrorResponse, GeographicBounds, 
    InfrastructureElement, AnalysisSummary, ImageMetadata, InfrastructureType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LaneSegNet Test API", 
    version="1.0.0-test",
    description="Minimal test server for validation without model dependencies"
)

# Mock coordinate transformer for testing
class MockCoordinateTransformer:
    def __init__(self, bounds, image_width, image_height, resolution_mpp):
        self.bounds = bounds
        self.image_width = image_width
        self.image_height = image_height
        self.resolution_mpp = resolution_mpp
    
    def transform_polyline_to_geographic(self, pixel_points):
        # Mock transformation - returns fake lat/lon points
        return [{"latitude": -27.4700 + i*0.0001, "longitude": 153.0250 + i*0.0001} 
                for i in range(len(pixel_points))]
    
    def pixel_distance_to_meters(self, pixel_distance):
        return pixel_distance * self.resolution_mpp
    
    def calculate_area_sqm(self, area_pixels):
        return area_pixels * (self.resolution_mpp ** 2)

# Mock imagery manager
class MockImageryManager:
    async def acquire_imagery(self, bounds, resolution_mpp, preferred_provider=None):
        # Return mock image data
        mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return {
            "image": mock_image,
            "metadata": {
                "source": f"Mock {preferred_provider or 'local'}",
                "resolution_mpp": resolution_mpp,
                "bounds": bounds
            }
        }

# Mock inference functions
def mock_run_inference(model, image_np):
    """Mock inference that returns a simple segmentation map."""
    height, width = image_np.shape[:2]
    # Create mock segmentation with some lane markings
    seg_map = np.zeros((height, width), dtype=np.uint8)
    
    # Add some mock lane markings
    seg_map[height//2-2:height//2+2, width//4:3*width//4] = 1  # white solid
    seg_map[height//3-1:height//3+1, width//6:5*width//6] = 2  # white dashed
    
    return seg_map

def mock_format_results(segmentation_map, original_shape):
    """Mock result formatting."""
    lane_markings = [
        {
            "class": "single_white_solid",
            "class_id": 1,
            "points": [[100, 256], [400, 256], [400, 260], [100, 260]],
            "confidence": 0.95,
            "area": 1200.0,
            "length": 300.0
        },
        {
            "class": "single_white_dashed", 
            "class_id": 2,
            "points": [[80, 170], [420, 170], [420, 172], [80, 172]],
            "confidence": 0.88,
            "area": 680.0,
            "length": 340.0
        }
    ]
    
    return {
        "lane_markings": lane_markings,
        "class_summary": {"single_white_solid": 1, "single_white_dashed": 1},
        "total_segments": 2
    }

# Initialize mock services
imagery_manager = MockImageryManager()

@app.post("/analyze_road_infrastructure", 
          response_model=RoadInfrastructureResponse,
          summary="Test Road Infrastructure Analysis",
          description="Mock implementation for testing API structure and coordinate transformation")
async def analyze_road_infrastructure(
    coordinates: GeographicBounds,
    analysis_type: str = "comprehensive",
    resolution: float = 0.1,
    visualize: bool = False,
    model_type: str = "mock"
):
    start_time = time.time()
    
    try:
        # Validate coordinate bounds
        if coordinates.north <= coordinates.south or coordinates.east <= coordinates.west:
            raise HTTPException(status_code=400, detail="Invalid coordinate bounds")
        
        logger.info(f"Mock analysis for coordinates: {coordinates}")
        
        # Mock imagery acquisition
        imagery_result = await imagery_manager.acquire_imagery(
            bounds=coordinates,
            resolution_mpp=resolution,
            preferred_provider="mock"
        )
        
        img_np_rgb = imagery_result["image"]
        image_metadata = imagery_result["metadata"]
        original_shape = (img_np_rgb.shape[1], img_np_rgb.shape[0])
        
        # Mock inference
        segmentation_map = mock_run_inference(None, img_np_rgb)
        detection_results = mock_format_results(segmentation_map, original_shape)
        
        # Mock coordinate transformation
        transformer = MockCoordinateTransformer(
            bounds=coordinates,
            image_width=original_shape[0],
            image_height=original_shape[1], 
            resolution_mpp=resolution
        )
        
        # Transform results
        infrastructure_elements = []
        lane_markings = detection_results.get("lane_markings", [])
        
        for marking in lane_markings:
            pixel_points = marking.get("points", [])
            geographic_points = transformer.transform_polyline_to_geographic(pixel_points)
            
            length_pixels = marking.get("length", 0)
            length_meters = transformer.pixel_distance_to_meters(length_pixels)
            area_pixels = marking.get("area", 0) 
            area_sqm = transformer.calculate_area_sqm(area_pixels)
            
            element = InfrastructureElement(
                **{"class": marking.get("class", "unknown")},  # Use class field with alias
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
        
        # Create response
        class_counts = detection_results.get("class_summary", {})
        analysis_summary = AnalysisSummary(
            total_elements=len(infrastructure_elements),
            elements_by_type={"lane_marking": len(infrastructure_elements)},
            elements_by_class=class_counts,
            total_lane_length_m=sum(e.length_meters or 0 for e in infrastructure_elements)
        )
        
        img_metadata = ImageMetadata(
            width=original_shape[0],
            height=original_shape[1],
            resolution_mpp=resolution,
            bounds=coordinates,
            source="Mock Testing"
        )
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        response = RoadInfrastructureResponse(
            infrastructure_elements=infrastructure_elements,
            analysis_summary=analysis_summary,
            image_metadata=img_metadata,
            processing_time_ms=processing_time_ms,
            model_type=f"mock-{model_type}",
            analysis_type=analysis_type,
            confidence_threshold=0.5,
            coverage_percentage=100.0
        )
        
        logger.info(f"Mock analysis completed in {processing_time_ms:.2f} ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Mock analysis failed")
        raise HTTPException(status_code=500, detail=f"Mock analysis failed: {e}")

@app.get("/health")
async def health_check():
    """Health check for test server."""
    return {
        "status": "ok", 
        "mode": "test",
        "model_loaded": "mock",
        "message": "Test server running without model dependencies"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)