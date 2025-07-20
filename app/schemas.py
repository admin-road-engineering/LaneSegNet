from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict

class LaneMarking(BaseModel):
    """Represents a detected lane marking segment with classification."""
    
    class_name: str = Field(..., alias="class", description="Lane marking class (e.g., single_white_solid, center_line)")
    class_id: int = Field(..., description="Numeric class identifier")
    points: List[List[float]] = Field(..., description="List of [x, y] coordinates defining the lane marking polyline")
    confidence: float = Field(default=1.0, description="Detection confidence score")
    area: float = Field(..., description="Area of the detected lane marking in pixels")
    length: float = Field(..., description="Perimeter/length of the lane marking in pixels")

    class Config:
        allow_population_by_field_name = True

class ImageShape(BaseModel):
    """Represents image dimensions."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")

class LaneDetectionResponse(BaseModel):
    """Response schema for lane marking detection API."""
    
    lane_markings: List[LaneMarking] = Field(..., description="List of detected lane marking segments")
    class_summary: Dict[str, int] = Field(..., description="Count of detections per lane marking class")
    total_segments: int = Field(..., description="Total number of detected lane marking segments")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_type: str = Field(..., description="Model used for detection (swin or lanesegnet)")
    image_shape: ImageShape = Field(..., description="Original image dimensions")

class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str = Field(..., description="Description of the error") 