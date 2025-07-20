"""
Simple API validation test
"""

from app.schemas import (
    GeographicBounds, InfrastructureElement, InfrastructureType,
    AnalysisSummary, ImageMetadata, RoadInfrastructureResponse
)

def test_basic_validation():
    print("Testing basic API schema validation...")
    
    # Test basic coordinate bounds
    bounds = GeographicBounds(
        north=-27.4698,
        south=-27.4705,
        east=153.0258,
        west=153.0251
    )
    print(f"PASS: Geographic bounds created: {bounds.north}, {bounds.south}")
    
    # Test infrastructure element creation
    element = InfrastructureElement(
        **{"class": "single_white_solid"},
        class_id=1,
        infrastructure_type=InfrastructureType.LANE_MARKING,
        points=[[100, 200], [300, 200]],
        confidence=0.95,
        area_pixels=1000.0,
        length_pixels=200.0
    )
    print(f"PASS: Infrastructure element created: {element.class_name}")
    
    # Test complete API response
    analysis_summary = AnalysisSummary(
        total_elements=1,
        elements_by_type={"lane_marking": 1},
        elements_by_class={"single_white_solid": 1},
        total_lane_length_m=20.0
    )
    
    img_metadata = ImageMetadata(
        width=512,
        height=512,
        resolution_mpp=0.1,
        bounds=bounds,
        source="Test"
    )
    
    response = RoadInfrastructureResponse(
        infrastructure_elements=[element],
        analysis_summary=analysis_summary,
        image_metadata=img_metadata,
        processing_time_ms=123.45,
        model_type="test",
        analysis_type="comprehensive"
    )
    
    print(f"PASS: Complete API response created with {len(response.infrastructure_elements)} elements")
    print(f"      Processing time: {response.processing_time_ms} ms")
    
    return True

if __name__ == "__main__":
    try:
        success = test_basic_validation()
        if success:
            print("\nSUCCESS: All API schema tests passed!")
            print("The API structure is validated and ready for testing.")
        else:
            print("\nFAILED: Some tests failed")
    except Exception as e:
        print(f"\nERROR: Test crashed: {e}")
        import traceback
        traceback.print_exc()