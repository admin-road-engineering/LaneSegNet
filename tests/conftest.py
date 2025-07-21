import pytest
import asyncio
import numpy as np
from PIL import Image
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from typing import Dict, Any, List

from app.main import app
from app.schemas import GeographicBounds, InfrastructureElement, InfrastructureType


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """FastAPI test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_model():
    """Mock segmentation model."""
    mock = Mock()
    mock.return_value = np.random.randint(0, 12, (512, 512), dtype=np.uint8)
    return mock


@pytest.fixture
def test_coordinates():
    """Standard test coordinates (Brisbane CBD area)."""
    return GeographicBounds(
        north=-27.4698,
        south=-27.4705,
        east=153.0258,
        west=153.0251
    )


@pytest.fixture
def test_image_rgb():
    """Test RGB image as numpy array."""
    return np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)


@pytest.fixture
def test_image_file():
    """Create a temporary test image file."""
    img = Image.new('RGB', (512, 512), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def mock_segmentation_map():
    """Mock segmentation map with lane markings."""
    # Create a segmentation map with some lane marking pixels
    seg_map = np.zeros((512, 512), dtype=np.uint8)
    # Add some lane marking patterns (class 6 is typically road in ADE20K)
    seg_map[100:110, 50:400] = 6  # Horizontal lane marking
    seg_map[200:210, 50:400] = 6  # Another horizontal lane marking
    return seg_map


@pytest.fixture
def mock_lane_markings():
    """Mock lane markings detection results."""
    return [
        {
            "class": "single_white_solid",
            "class_id": 1,
            "confidence": 0.85,
            "points": [[100, 50], [400, 50], [400, 60], [100, 60]],
            "area_pixels": 3000,
            "length_pixels": 300
        },
        {
            "class": "single_yellow_solid", 
            "class_id": 2,
            "confidence": 0.92,
            "points": [[100, 200], [400, 200], [400, 210], [100, 210]],
            "area_pixels": 3000,
            "length_pixels": 300
        }
    ]


@pytest.fixture
def mock_imagery_result():
    """Mock imagery acquisition result."""
    return {
        "image": np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8),
        "metadata": {
            "source": "local_imagery",
            "resolution_mpp": 0.1,
            "acquisition_date": "2024-01-15"
        }
    }


@pytest.fixture
def mock_detection_results():
    """Mock complete detection results."""
    return {
        "lane_markings": [
            {
                "class": "single_white_solid",
                "class_id": 1,
                "confidence": 0.85,
                "points": [[100, 50], [400, 50]],
                "area_pixels": 3000,
                "length_pixels": 300
            }
        ],
        "class_summary": {
            "single_white_solid": 1,
            "total": 1
        },
        "processing_info": {
            "total_pixels": 1638400,
            "road_pixels": 500000
        }
    }


@pytest.fixture
def mock_infrastructure_elements():
    """Mock infrastructure elements for testing."""
    return [
        InfrastructureElement(
            **{"class": "single_white_solid"},
            class_id=1,
            infrastructure_type=InfrastructureType.LANE_MARKING,
            points=[[100, 50], [400, 50]],
            geographic_points=None,
            confidence=0.85,
            area_pixels=3000,
            area_sqm=30.0,
            length_pixels=300,
            length_meters=30.0
        )
    ]


@pytest.fixture
async def mock_imagery_manager():
    """Mock imagery manager."""
    manager = AsyncMock()
    manager.acquire_imagery.return_value = {
        "image": np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8),
        "metadata": {
            "source": "local_imagery",
            "resolution_mpp": 0.1
        }
    }
    return manager


@pytest.fixture
def mock_coordinate_transformer():
    """Mock coordinate transformer."""
    transformer = Mock()
    transformer.transform_polyline_to_geographic.return_value = [
        {"latitude": -27.4701, "longitude": 153.0254},
        {"latitude": -27.4701, "longitude": 153.0255}
    ]
    transformer.pixel_distance_to_meters.return_value = 30.0
    transformer.calculate_area_sqm.return_value = 30.0
    return transformer


# Patch decorators for common mocks
@pytest.fixture
def patch_model_loading():
    """Patch model loading to return a mock model."""
    with patch('app.model_loader.load_model') as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def patch_run_inference():
    """Patch inference function."""
    with patch('app.inference.run_inference') as mock:
        mock.return_value = np.random.randint(0, 12, (512, 512), dtype=np.uint8)
        yield mock


@pytest.fixture
def patch_format_results():
    """Patch format_results function."""
    with patch('app.inference.format_results') as mock:
        mock.return_value = {
            "lane_markings": [
                {
                    "class": "single_white_solid",
                    "class_id": 1,
                    "confidence": 0.85,
                    "points": [[100, 50], [400, 50]]
                }
            ],
            "class_summary": {"single_white_solid": 1}
        }
        yield mock


@pytest.fixture
def patch_imagery_manager():
    """Patch imagery manager."""
    async def mock_acquire_imagery(*args, **kwargs):
        return {
            "image": np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8),
            "metadata": {"source": "test", "resolution_mpp": 0.1}
        }
    
    with patch('app.main.imagery_manager') as mock:
        mock.acquire_imagery = AsyncMock(side_effect=mock_acquire_imagery)
        yield mock


# Test data generators
def generate_test_coordinates(offset: float = 0.0) -> GeographicBounds:
    """Generate test coordinates with optional offset."""
    return GeographicBounds(
        north=-27.4698 + offset,
        south=-27.4705 + offset,
        east=153.0258 + offset,
        west=153.0251 + offset
    )


def generate_mock_segmentation_map(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate a mock segmentation map."""
    seg_map = np.zeros((height, width), dtype=np.uint8)
    # Add road pixels (class 6)
    seg_map[height//4:3*height//4, width//4:3*width//4] = 6
    return seg_map


def generate_test_image(width: int = 512, height: int = 512, channels: int = 3) -> np.ndarray:
    """Generate a test image."""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


# Custom assertions
def assert_valid_coordinates(coords: GeographicBounds):
    """Assert that coordinates are valid."""
    assert coords.north > coords.south, "North must be greater than South"
    assert coords.east > coords.west, "East must be greater than West"
    assert -90 <= coords.south <= 90, "South latitude out of range"
    assert -90 <= coords.north <= 90, "North latitude out of range"
    assert -180 <= coords.west <= 180, "West longitude out of range"
    assert -180 <= coords.east <= 180, "East longitude out of range"


def assert_valid_infrastructure_element(element: InfrastructureElement):
    """Assert that an infrastructure element is valid."""
    assert element.class_name is not None, "Class name must be provided"
    assert element.class_id >= 0, "Class ID must be non-negative"
    assert 0 <= element.confidence <= 1, "Confidence must be between 0 and 1"
    assert len(element.points) >= 2, "Must have at least 2 points"
    assert element.area_pixels >= 0, "Area in pixels must be non-negative"
    assert element.length_pixels >= 0, "Length in pixels must be non-negative"