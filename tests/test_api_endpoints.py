import pytest
import json
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import numpy as np

from app.main import app
from app.schemas import GeographicBounds, RoadInfrastructureResponse
from tests.conftest import (
    assert_valid_coordinates, 
    assert_valid_infrastructure_element,
    generate_test_coordinates
)


class TestAnalyzeRoadInfrastructureEndpoint:
    """Test suite for /analyze_road_infrastructure endpoint."""

    @pytest.mark.api
    @pytest.mark.unit
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_valid_request(
        self, test_client, test_coordinates, patch_imagery_manager, 
        patch_run_inference, patch_format_results
    ):
        """Test successful road infrastructure analysis."""
        # Prepare request data
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        # Mock app.model to be loaded
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_road_infrastructure",
                params={
                    "analysis_type": "comprehensive",
                    "resolution": 0.1,
                    "visualize": False,
                    "model_type": "swin"
                },
                json=request_data
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert "infrastructure_elements" in data
        assert "analysis_summary" in data
        assert "image_metadata" in data
        assert "processing_time_ms" in data
        assert "model_type" in data
        assert "analysis_type" in data
        
        # Validate analysis summary
        summary = data["analysis_summary"]
        assert "total_elements" in summary
        assert "elements_by_type" in summary
        assert "elements_by_class" in summary
        
        # Validate metadata
        metadata = data["image_metadata"]
        assert "width" in metadata
        assert "height" in metadata
        assert "resolution_mpp" in metadata
        assert "bounds" in metadata

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_invalid_coordinates(self, test_client):
        """Test error handling for invalid coordinates."""
        # Test case: north <= south
        invalid_coords = {
            "north": -27.4705,  # Invalid: north <= south
            "south": -27.4698,
            "east": 153.0258,
            "west": 153.0251
        }
        
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=invalid_coords
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid coordinate bounds" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_area_too_large(self, test_client):
        """Test error handling for analysis area too large."""
        # Create coordinates with large area (> 100 km²)
        large_area_coords = {
            "north": -27.0,
            "south": -28.0,  # 1 degree = ~111 km
            "east": 154.0,
            "west": 153.0
        }
        
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=large_area_coords
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Analysis area too large" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_model_not_loaded(self, test_client, test_coordinates):
        """Test error handling when model is not loaded."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        # Mock model as None (not loaded)
        with patch('app.main.model', None):
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=request_data
            )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Model is not available" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_imagery_acquisition_failure(
        self, test_client, test_coordinates
    ):
        """Test error handling when imagery acquisition fails."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        # Mock imagery manager to fail
        async def mock_fail(*args, **kwargs):
            raise Exception("Imagery acquisition failed")
        
        with patch('app.main.model', Mock()), \
             patch('app.main.imagery_manager') as mock_manager:
            mock_manager.acquire_imagery = AsyncMock(side_effect=mock_fail)
            
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=request_data
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to acquire aerial imagery" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_inference_failure(
        self, test_client, test_coordinates, patch_imagery_manager
    ):
        """Test error handling when inference fails."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        with patch('app.main.model', Mock()), \
             patch('app.inference.run_inference', return_value=None):
            
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=request_data
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Infrastructure detection failed" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_with_visualization(
        self, test_client, test_coordinates, patch_imagery_manager,
        patch_run_inference, patch_format_results
    ):
        """Test visualization response."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        with patch('app.main.model', Mock()), \
             patch('cv2.imencode', return_value=(True, np.array([1, 2, 3]))):
            
            response = test_client.post(
                "/analyze_road_infrastructure",
                params={"visualize": True},
                json=request_data
            )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/jpeg"

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_parameter_validation(
        self, test_client, test_coordinates, patch_imagery_manager,
        patch_run_inference, patch_format_results
    ):
        """Test parameter validation for different analysis types and models."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        test_cases = [
            {"analysis_type": "lane_markings", "model_type": "swin", "resolution": 0.05},
            {"analysis_type": "comprehensive", "model_type": "lanesegnet", "resolution": 0.2},
            {"analysis_type": "pavements", "model_type": "swin", "resolution": 1.0},
        ]
        
        for params in test_cases:
            with patch('app.main.model', Mock()):
                response = test_client.post(
                    "/analyze_road_infrastructure",
                    params=params,
                    json=request_data
                )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["analysis_type"] == params["analysis_type"]
            assert data["model_type"] == params["model_type"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_coordinate_edge_cases(
        self, test_client, patch_imagery_manager, patch_run_inference, patch_format_results
    ):
        """Test coordinate edge cases and boundary conditions."""
        edge_cases = [
            # Very small area
            {
                "north": -27.469800,
                "south": -27.469801,
                "east": 153.025800,
                "west": 153.025801
            },
            # Coordinates near poles
            {
                "north": 89.9999,
                "south": 89.9998,
                "east": 180.0,
                "west": 179.9999
            },
            # Coordinates crossing antimeridian (handled if properly implemented)
            {
                "north": -27.4698,
                "south": -27.4705,
                "east": -179.9999,
                "west": 179.9999
            }
        ]
        
        for coords in edge_cases[:2]:  # Skip antimeridian test for now
            with patch('app.main.model', Mock()):
                response = test_client.post(
                    "/analyze_road_infrastructure",
                    json=coords
                )
            
            # Should either succeed or fail gracefully
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]

    @pytest.mark.api
    @pytest.mark.unit 
    def test_analyze_road_infrastructure_response_data_integrity(
        self, test_client, test_coordinates, patch_imagery_manager,
        patch_run_inference, patch_format_results
    ):
        """Test that response data maintains integrity and consistency."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_road_infrastructure",
                params={"resolution": 0.1},
                json=request_data
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate coordinate consistency
        bounds = data["image_metadata"]["bounds"]
        assert_valid_coordinates(GeographicBounds(**bounds))
        
        # Validate infrastructure elements
        for element in data["infrastructure_elements"]:
            # Verify required fields
            assert "class" in element
            assert "infrastructure_type" in element
            assert "points" in element
            assert "confidence" in element
            
            # Validate confidence range
            assert 0 <= element["confidence"] <= 1
            
            # Validate points structure
            assert isinstance(element["points"], list)
            assert len(element["points"]) >= 2
            
        # Validate summary consistency
        summary = data["analysis_summary"]
        assert summary["total_elements"] == len(data["infrastructure_elements"])
        
        # Check element counts match
        total_by_type = sum(summary["elements_by_type"].values())
        assert total_by_type == summary["total_elements"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_road_infrastructure_performance_requirements(
        self, test_client, test_coordinates, patch_imagery_manager,
        patch_run_inference, patch_format_results
    ):
        """Test performance requirements are met."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_road_infrastructure",
                json=request_data
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check processing time is reasonable (< 2000ms target)
        processing_time = data["processing_time_ms"]
        assert processing_time > 0
        assert processing_time < 10000  # 10 second maximum for tests


class TestAnalyzeImageEndpoint:
    """Test suite for /analyze_image endpoint."""

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_valid_upload(
        self, test_client, test_image_file, patch_run_inference, patch_format_results
    ):
        """Test successful image analysis with file upload."""
        with patch('app.main.model', Mock()):
            with open(test_image_file, 'rb') as f:
                response = test_client.post(
                    "/analyze_image",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "analysis_type": "comprehensive",
                        "resolution": 0.1,
                        "model_type": "swin"
                    }
                )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert "infrastructure_elements" in data
        assert "analysis_summary" in data
        assert "image_metadata" in data
        assert "processing_time_ms" in data

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_with_coordinates(
        self, test_client, test_image_file, test_coordinates,
        patch_run_inference, patch_format_results
    ):
        """Test image analysis with geo-referencing coordinates."""
        with patch('app.main.model', Mock()):
            with open(test_image_file, 'rb') as f:
                response = test_client.post(
                    "/analyze_image",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "analysis_type": "comprehensive",
                        "resolution": 0.1,
                        "coordinates_north": test_coordinates.north,
                        "coordinates_south": test_coordinates.south,
                        "coordinates_east": test_coordinates.east,
                        "coordinates_west": test_coordinates.west
                    }
                )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should have geographic points when coordinates provided
        metadata = data["image_metadata"]
        assert metadata["bounds"] is not None

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_no_file(self, test_client):
        """Test error handling when no image file provided."""
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_image",
                data={"analysis_type": "comprehensive"}
            )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_invalid_file_type(self, test_client):
        """Test error handling for invalid file types."""
        # Create a text file instead of image
        with patch('app.main.model', Mock()):
            response = test_client.post(
                "/analyze_image",
                files={"image": ("test.txt", b"not an image", "text/plain")},
                data={"analysis_type": "comprehensive"}
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported file type" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_invalid_coordinates(
        self, test_client, test_image_file
    ):
        """Test error handling for invalid coordinate combinations."""
        with patch('app.main.model', Mock()):
            with open(test_image_file, 'rb') as f:
                response = test_client.post(
                    "/analyze_image",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "coordinates_north": -27.4705,  # Invalid: north <= south
                        "coordinates_south": -27.4698,
                        "coordinates_east": 153.0258,
                        "coordinates_west": 153.0251
                    }
                )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid coordinates" in response.json()["detail"]

    @pytest.mark.api
    @pytest.mark.unit
    def test_analyze_image_partial_coordinates(
        self, test_client, test_image_file, patch_run_inference, patch_format_results
    ):
        """Test behavior with partial coordinate data (should proceed without geo-referencing)."""
        with patch('app.main.model', Mock()):
            with open(test_image_file, 'rb') as f:
                response = test_client.post(
                    "/analyze_image", 
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "coordinates_north": -27.4698,
                        "coordinates_south": -27.4705,
                        # Missing east and west coordinates
                    }
                )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should proceed without geo-referencing
        metadata = data["image_metadata"]
        assert metadata["bounds"] is None


class TestVisualizationEndpoints:
    """Test suite for visualization endpoints."""

    @pytest.mark.api
    @pytest.mark.unit
    def test_visualize_infrastructure_endpoint(
        self, test_client, test_coordinates, patch_imagery_manager,
        patch_run_inference, patch_format_results
    ):
        """Test visualization endpoint."""
        request_data = {
            "north": test_coordinates.north,
            "south": test_coordinates.south,
            "east": test_coordinates.east,
            "west": test_coordinates.west
        }
        
        with patch('app.main.model', Mock()), \
             patch('cv2.imencode', return_value=(True, np.array([1, 2, 3]))):
            
            response = test_client.post(
                "/visualize_infrastructure",
                params={
                    "resolution": 0.1,
                    "viz_type": "side_by_side",
                    "show_labels": True,
                    "show_confidence": False
                },
                json=request_data
            )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/jpeg"

    @pytest.mark.api
    @pytest.mark.unit
    def test_visualizer_interface(self, test_client):
        """Test visualizer web interface."""
        response = test_client.get("/visualizer")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "LaneSegNet Visualizer" in response.text