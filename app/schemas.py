from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Literal
from enum import Enum

class InfrastructureType(str, Enum):
    """Enumeration of road infrastructure types."""
    LANE_MARKING = "lane_marking"
    ROAD_SURFACE = "road_surface"
    PEDESTRIAN = "pedestrian_infrastructure"
    ROAD_FEATURE = "road_feature"
    UTILITY = "utility"
    BOUNDARY = "boundary"

class GeographicBounds(BaseModel):
    """Geographic bounding box coordinates."""
    north: float = Field(..., description="Northern boundary latitude")
    south: float = Field(..., description="Southern boundary latitude") 
    east: float = Field(..., description="Eastern boundary longitude")
    west: float = Field(..., description="Western boundary longitude")

class GeographicPoint(BaseModel):
    """Geographic coordinate point."""
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")

class InfrastructureElement(BaseModel):
    """Represents a detected road infrastructure element."""
    
    class_name: str = Field(..., alias="class", description="Infrastructure class name")
    class_id: int = Field(..., description="Numeric class identifier")
    infrastructure_type: InfrastructureType = Field(..., description="Type of infrastructure")
    
    # Geometric data
    points: List[List[float]] = Field(..., description="Pixel coordinates [x, y] defining the element")
    geographic_points: Optional[List[GeographicPoint]] = Field(None, description="Geographic coordinates (lat/lon)")
    
    # Properties
    confidence: float = Field(default=1.0, description="Detection confidence score (0-1)")
    area_pixels: float = Field(..., description="Area in pixels")
    area_sqm: Optional[float] = Field(None, description="Area in square meters")
    length_pixels: float = Field(..., description="Length/perimeter in pixels")
    length_meters: Optional[float] = Field(None, description="Length in meters")
    
    # Additional attributes
    condition: Optional[str] = Field(None, description="Condition assessment (good, fair, poor)")
    material: Optional[str] = Field(None, description="Material type (asphalt, concrete, etc.)")
    width_meters: Optional[float] = Field(None, description="Width measurement in meters")

    class Config:
        allow_population_by_field_name = True

class ImageMetadata(BaseModel):
    """Metadata about the analyzed aerial imagery."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    resolution_mpp: float = Field(..., description="Meters per pixel resolution")
    bounds: GeographicBounds = Field(..., description="Geographic bounds of the image")
    acquisition_date: Optional[str] = Field(None, description="Image acquisition date")
    source: Optional[str] = Field(None, description="Imagery source (satellite, drone, etc.)")

class AnalysisSummary(BaseModel):
    """Summary statistics of the infrastructure analysis."""
    total_elements: int = Field(..., description="Total number of detected elements")
    elements_by_type: Dict[str, int] = Field(..., description="Count by infrastructure type")
    elements_by_class: Dict[str, int] = Field(..., description="Count by specific class")
    total_road_area_sqm: Optional[float] = Field(None, description="Total road area analyzed")
    total_lane_length_m: Optional[float] = Field(None, description="Total lane marking length")

class RoadInfrastructureResponse(BaseModel):
    """Response schema for comprehensive road infrastructure analysis."""
    
    # Core results
    infrastructure_elements: List[InfrastructureElement] = Field(..., description="All detected infrastructure elements")
    analysis_summary: AnalysisSummary = Field(..., description="Summary statistics")
    
    # Metadata
    image_metadata: ImageMetadata = Field(..., description="Imagery and analysis metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_type: str = Field(..., description="Model used for analysis")
    analysis_type: str = Field(..., description="Type of analysis performed")
    
    # Quality metrics
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence threshold applied")
    coverage_percentage: Optional[float] = Field(None, description="Percentage of area successfully analyzed")

class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str = Field(..., description="Description of the error") 