from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class LaneSegment(BaseModel):
    points: List[Tuple[float, float]] = Field(..., description="List of [x, y] coordinates defining the lane segment polyline in image pixel coordinates.")
    confidence: Optional[float] = Field(None, description="Model's confidence score for the detection.")
    type: Optional[str] = Field(None, description="Type of lane marking (e.g., solid, dashed, boundary).")

class LaneDetectionResponse(BaseModel):
    lanes: List[LaneSegment] = Field(..., description="A list of detected lane segments.")
    processing_time_ms: Optional[float] = Field(None, description="Time taken for inference in milliseconds.")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Description of the error.") 