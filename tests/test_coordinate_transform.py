import pytest
import numpy as np
import math
from unittest.mock import Mock

from app.coordinate_transform import CoordinateTransformer, GeographicAnalyzer
from app.schemas import GeographicBounds, GeographicPoint
from tests.conftest import assert_valid_coordinates


class TestCoordinateTransformer:
    """Test suite for CoordinateTransformer class."""

    @pytest.fixture
    def brisbane_bounds(self):
        """Brisbane test area bounds."""
        return GeographicBounds(
            north=-27.4698,
            south=-27.4705, 
            east=153.0258,
            west=153.0251
        )

    @pytest.fixture
    def transformer(self, brisbane_bounds):
        """CoordinateTransformer fixture with Brisbane bounds."""
        return CoordinateTransformer(
            bounds=brisbane_bounds,
            image_width=1280,
            image_height=1280,
            resolution_mpp=0.1
        )

    @pytest.mark.unit
    def test_transformer_initialization(self, transformer, brisbane_bounds):
        """Test coordinate transformer initialization."""
        assert transformer.bounds == brisbane_bounds
        assert transformer.image_width == 1280
        assert transformer.image_height == 1280
        assert transformer.resolution_mpp == 0.1
        
        # Check calculated parameters
        assert transformer.lat_extent > 0
        assert transformer.lon_extent > 0
        assert transformer.pixels_per_lat_degree > 0
        assert transformer.pixels_per_lon_degree > 0
        assert transformer.meters_per_lat_degree > 0
        assert transformer.meters_per_lon_degree > 0

    @pytest.mark.unit
    def test_geographic_to_pixel_center(self, transformer, brisbane_bounds):
        """Test geographic to pixel conversion for center point."""
        # Center of the bounds
        center_lat = (brisbane_bounds.north + brisbane_bounds.south) / 2
        center_lon = (brisbane_bounds.east + brisbane_bounds.west) / 2
        
        x, y = transformer.geographic_to_pixel(center_lat, center_lon)
        
        # Should be approximately at image center
        assert abs(x - 640) < 10  # Within 10 pixels of center
        assert abs(y - 640) < 10

    @pytest.mark.unit
    def test_geographic_to_pixel_corners(self, transformer, brisbane_bounds):
        """Test geographic to pixel conversion for corner points."""
        # Test corners
        corners = [
            (brisbane_bounds.north, brisbane_bounds.west),  # Top-left
            (brisbane_bounds.north, brisbane_bounds.east),  # Top-right
            (brisbane_bounds.south, brisbane_bounds.west),  # Bottom-left
            (brisbane_bounds.south, brisbane_bounds.east),  # Bottom-right
        ]
        
        expected_pixels = [
            (0, 0),      # Top-left in geographic = top-left in image
            (1279, 0),   # Top-right in geographic = top-right in image
            (0, 1279),   # Bottom-left in geographic = bottom-left in image
            (1279, 1279) # Bottom-right in geographic = bottom-right in image
        ]
        
        for (lat, lon), (exp_x, exp_y) in zip(corners, expected_pixels):
            x, y = transformer.geographic_to_pixel(lat, lon)
            
            # Allow small tolerance due to rounding
            assert abs(x - exp_x) <= 1
            assert abs(y - exp_y) <= 1

    @pytest.mark.unit
    def test_pixel_to_geographic_center(self, transformer, brisbane_bounds):
        """Test pixel to geographic conversion for center point."""
        # Center pixel
        lat, lon = transformer.pixel_to_geographic(640, 640)
        
        # Should be approximately at geographic center
        center_lat = (brisbane_bounds.north + brisbane_bounds.south) / 2
        center_lon = (brisbane_bounds.east + brisbane_bounds.west) / 2
        
        assert abs(lat - center_lat) < 0.0001  # Very small tolerance
        assert abs(lon - center_lon) < 0.0001

    @pytest.mark.unit
    def test_pixel_to_geographic_corners(self, transformer, brisbane_bounds):
        """Test pixel to geographic conversion for corner pixels."""
        corners = [
            (0, 0),        # Top-left pixel
            (1279, 0),     # Top-right pixel
            (0, 1279),     # Bottom-left pixel
            (1279, 1279)   # Bottom-right pixel
        ]
        
        expected_coords = [
            (brisbane_bounds.north, brisbane_bounds.west),  # Top-left
            (brisbane_bounds.north, brisbane_bounds.east),  # Top-right
            (brisbane_bounds.south, brisbane_bounds.west),  # Bottom-left
            (brisbane_bounds.south, brisbane_bounds.east),  # Bottom-right
        ]
        
        for (x, y), (exp_lat, exp_lon) in zip(corners, expected_coords):
            lat, lon = transformer.pixel_to_geographic(x, y)
            
            # Allow small tolerance due to pixel discretization
            assert abs(lat - exp_lat) < 0.0001
            assert abs(lon - exp_lon) < 0.0001

    @pytest.mark.unit
    def test_round_trip_conversion(self, transformer):
        """Test round-trip conversion accuracy."""
        # Test various points
        test_coords = [
            (-27.4700, 153.0254),  # Near center
            (-27.4699, 153.0252),  # Offset point
            (-27.4703, 153.0256),  # Another offset point
        ]
        
        for original_lat, original_lon in test_coords:
            # Geographic -> Pixel -> Geographic
            x, y = transformer.geographic_to_pixel(original_lat, original_lon)
            converted_lat, converted_lon = transformer.pixel_to_geographic(x, y)
            
            # Should be very close to original
            assert abs(converted_lat - original_lat) < 0.001
            assert abs(converted_lon - original_lon) < 0.001

    @pytest.mark.unit
    def test_pixel_distance_to_meters(self, transformer):
        """Test pixel distance to meters conversion."""
        # Test known distances
        test_distances = [10, 100, 500, 1000]  # pixels
        
        for pixel_dist in test_distances:
            meters = transformer.pixel_distance_to_meters(pixel_dist)
            
            # Should be positive and reasonable
            assert meters > 0
            assert meters == pixel_dist * transformer.resolution_mpp

    @pytest.mark.unit
    def test_calculate_area_sqm(self, transformer):
        """Test area calculation in square meters."""
        # Test various pixel areas
        test_areas = [100, 1000, 10000]  # square pixels
        
        for pixel_area in test_areas:
            sqm = transformer.calculate_area_sqm(pixel_area)
            
            # Should be positive and reasonable
            assert sqm > 0
            expected_sqm = pixel_area * (transformer.resolution_mpp ** 2)
            assert abs(sqm - expected_sqm) < 0.001

    @pytest.mark.unit
    def test_transform_polyline_to_geographic(self, transformer):
        """Test polyline transformation to geographic coordinates."""
        # Test polyline (rectangle in pixel coordinates)
        pixel_points = [
            [100, 100],
            [200, 100], 
            [200, 200],
            [100, 200]
        ]
        
        geo_points = transformer.transform_polyline_to_geographic(pixel_points)
        
        # Should have same number of points
        assert len(geo_points) == len(pixel_points)
        
        # Each point should be a valid GeographicPoint
        for point in geo_points:
            assert isinstance(point, GeographicPoint)
            assert -90 <= point.latitude <= 90
            assert -180 <= point.longitude <= 180

    @pytest.mark.unit
    def test_coordinate_bounds_validation(self, transformer, brisbane_bounds):
        """Test coordinate clamping to image bounds."""
        # Test coordinates outside bounds
        out_of_bounds_coords = [
            (brisbane_bounds.north + 0.1, brisbane_bounds.west),  # Too far north
            (brisbane_bounds.south - 0.1, brisbane_bounds.west),  # Too far south
            (brisbane_bounds.north, brisbane_bounds.east + 0.1),  # Too far east
            (brisbane_bounds.north, brisbane_bounds.west - 0.1),  # Too far west
        ]
        
        for lat, lon in out_of_bounds_coords:
            x, y = transformer.geographic_to_pixel(lat, lon)
            
            # Should be clamped to image bounds
            assert 0 <= x < transformer.image_width
            assert 0 <= y < transformer.image_height

    @pytest.mark.unit
    def test_different_image_sizes(self, brisbane_bounds):
        """Test transformer with different image sizes."""
        sizes = [(512, 512), (1024, 768), (2048, 2048)]
        
        for width, height in sizes:
            transformer = CoordinateTransformer(
                bounds=brisbane_bounds,
                image_width=width,
                image_height=height,
                resolution_mpp=0.1
            )
            
            # Test center conversion
            center_lat = (brisbane_bounds.north + brisbane_bounds.south) / 2
            center_lon = (brisbane_bounds.east + brisbane_bounds.west) / 2
            
            x, y = transformer.geographic_to_pixel(center_lat, center_lon)
            
            # Should be at approximately image center
            assert abs(x - width // 2) < width * 0.1
            assert abs(y - height // 2) < height * 0.1

    @pytest.mark.unit
    def test_different_resolutions(self, brisbane_bounds):
        """Test transformer with different resolutions."""
        resolutions = [0.05, 0.1, 0.5, 1.0]
        
        for resolution in resolutions:
            transformer = CoordinateTransformer(
                bounds=brisbane_bounds,
                image_width=1280,
                image_height=1280,
                resolution_mpp=resolution
            )
            
            # Test distance conversion
            pixel_dist = 100
            meters = transformer.pixel_distance_to_meters(pixel_dist)
            
            assert meters == pixel_dist * resolution

    @pytest.mark.unit
    def test_edge_coordinates(self, transformer, brisbane_bounds):
        """Test coordinates at the exact edges of bounds."""
        edge_coords = [
            (brisbane_bounds.north, brisbane_bounds.west),
            (brisbane_bounds.north, brisbane_bounds.east),
            (brisbane_bounds.south, brisbane_bounds.west),
            (brisbane_bounds.south, brisbane_bounds.east),
        ]
        
        for lat, lon in edge_coords:
            x, y = transformer.geographic_to_pixel(lat, lon)
            
            # Should be within image bounds
            assert 0 <= x < transformer.image_width
            assert 0 <= y < transformer.image_height
            
            # Round trip should be close
            lat_back, lon_back = transformer.pixel_to_geographic(x, y)
            assert abs(lat_back - lat) < 0.001
            assert abs(lon_back - lon) < 0.001


class TestGeographicAnalyzer:
    """Test suite for GeographicAnalyzer class."""

    @pytest.fixture
    def analyzer(self, transformer):
        """GeographicAnalyzer fixture."""
        return GeographicAnalyzer(transformer)

    @pytest.fixture
    def sample_polyline(self):
        """Sample polyline in pixel coordinates."""
        return [
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200],
            [100, 100]  # Closed polyline
        ]

    @pytest.mark.unit
    def test_analyzer_initialization(self, analyzer, transformer):
        """Test geographic analyzer initialization."""
        assert analyzer.transformer == transformer

    @pytest.mark.unit
    def test_calculate_polyline_length_meters(self, analyzer, sample_polyline):
        """Test polyline length calculation in meters."""
        length = analyzer.calculate_polyline_length_meters(sample_polyline)
        
        # Should be positive
        assert length > 0
        
        # For a square polyline, perimeter should be 4 * side_length
        side_length_pixels = 100
        expected_length = 4 * analyzer.transformer.pixel_distance_to_meters(side_length_pixels)
        
        # Allow some tolerance for floating point arithmetic
        assert abs(length - expected_length) < 1.0

    @pytest.mark.unit
    def test_calculate_polygon_area_sqm(self, analyzer, sample_polyline):
        """Test polygon area calculation in square meters."""
        area = analyzer.calculate_polygon_area_sqm(sample_polyline)
        
        # Should be positive
        assert area > 0
        
        # For a 100x100 pixel square
        expected_area_pixels = 100 * 100
        expected_area_sqm = analyzer.transformer.calculate_area_sqm(expected_area_pixels)
        
        # Allow some tolerance
        assert abs(area - expected_area_sqm) < expected_area_sqm * 0.1

    @pytest.mark.unit
    def test_get_polyline_bounds(self, analyzer, sample_polyline):
        """Test polyline bounds calculation."""
        bounds = analyzer.get_polyline_bounds(sample_polyline)
        
        # Should have correct structure
        assert 'min_lat' in bounds
        assert 'max_lat' in bounds
        assert 'min_lon' in bounds
        assert 'max_lon' in bounds
        
        # Bounds should be sensible
        assert bounds['min_lat'] <= bounds['max_lat']
        assert bounds['min_lon'] <= bounds['max_lon']

    @pytest.mark.unit
    def test_point_in_bounds(self, analyzer):
        """Test point-in-bounds checking."""
        bounds = analyzer.transformer.bounds
        
        # Point inside bounds
        center_lat = (bounds.north + bounds.south) / 2
        center_lon = (bounds.east + bounds.west) / 2
        assert analyzer.point_in_bounds(center_lat, center_lon)
        
        # Point outside bounds
        assert not analyzer.point_in_bounds(bounds.north + 1.0, center_lon)
        assert not analyzer.point_in_bounds(center_lat, bounds.west - 1.0)

    @pytest.mark.unit
    def test_calculate_centroid(self, analyzer, sample_polyline):
        """Test centroid calculation."""
        centroid = analyzer.calculate_centroid(sample_polyline)
        
        # Should be a GeographicPoint
        assert isinstance(centroid, GeographicPoint)
        
        # For a square centered at (150, 150), centroid should be near center
        center_lat, center_lon = analyzer.transformer.pixel_to_geographic(150, 150)
        
        assert abs(centroid.latitude - center_lat) < 0.001
        assert abs(centroid.longitude - center_lon) < 0.001

    @pytest.mark.unit
    def test_analyze_infrastructure_geometry(self, analyzer, sample_polyline):
        """Test comprehensive geometry analysis."""
        analysis = analyzer.analyze_infrastructure_geometry(sample_polyline)
        
        # Should have all required fields
        required_fields = [
            'length_meters', 'area_sqm', 'centroid', 'bounds',
            'perimeter_meters', 'aspect_ratio', 'complexity_score'
        ]
        
        for field in required_fields:
            assert field in analysis
        
        # Values should be reasonable
        assert analysis['length_meters'] > 0
        assert analysis['area_sqm'] > 0
        assert isinstance(analysis['centroid'], GeographicPoint)
        assert analysis['perimeter_meters'] > 0
        assert analysis['aspect_ratio'] > 0
        assert analysis['complexity_score'] >= 0

    @pytest.mark.unit
    def test_empty_polyline_handling(self, analyzer):
        """Test handling of empty polylines."""
        # Empty polyline
        with pytest.raises((ValueError, IndexError)):
            analyzer.calculate_polyline_length_meters([])
        
        # Single point
        with pytest.raises((ValueError, IndexError)):
            analyzer.calculate_polyline_length_meters([[100, 100]])

    @pytest.mark.unit
    def test_coordinate_precision(self, analyzer):
        """Test coordinate transformation precision."""
        # Test high-precision coordinates
        precise_points = [
            [100.123, 100.456],
            [200.789, 100.012],
            [200.345, 200.678]
        ]
        
        # Should handle fractional pixel coordinates
        length = analyzer.calculate_polyline_length_meters(precise_points)
        assert length > 0
        
        # Geographic conversion should maintain reasonable precision
        geo_points = analyzer.transformer.transform_polyline_to_geographic(precise_points)
        assert len(geo_points) == len(precise_points)


class TestCoordinateTransformIntegration:
    """Integration tests for coordinate transformation workflow."""

    @pytest.mark.integration
    def test_real_world_coordinate_accuracy(self):
        """Test coordinate transformation accuracy with real-world data."""
        # Brisbane area with known landmarks
        bounds = GeographicBounds(
            north=-27.4698,  # Brisbane CBD north
            south=-27.4705,  # Brisbane CBD south
            east=153.0258,   # Brisbane CBD east
            west=153.0251    # Brisbane CBD west
        )
        
        transformer = CoordinateTransformer(
            bounds=bounds,
            image_width=1280,
            image_height=1280,
            resolution_mpp=0.1
        )
        
        # Test known Brisbane coordinates
        brisbane_city_hall = (-27.4703, 153.0254)  # Approximate
        
        # Convert to pixel and back
        x, y = transformer.geographic_to_pixel(*brisbane_city_hall)
        lat_back, lon_back = transformer.pixel_to_geographic(x, y)
        
        # Should be very close to original
        assert abs(lat_back - brisbane_city_hall[0]) < 0.0001
        assert abs(lon_back - brisbane_city_hall[1]) < 0.0001
        
        # Should be within image bounds
        assert 0 <= x < 1280
        assert 0 <= y < 1280

    @pytest.mark.integration
    def test_multiple_coordinate_systems(self):
        """Test with different geographic coordinate systems."""
        test_locations = [
            # Brisbane, Australia
            GeographicBounds(north=-27.4698, south=-27.4705, east=153.0258, west=153.0251),
            # New York, USA
            GeographicBounds(north=40.7831, south=40.7489, east=-73.9441, west=-74.0059),
            # London, UK
            GeographicBounds(north=51.5074, south=51.4994, east=-0.1278, west=-0.1419),
            # Tokyo, Japan
            GeographicBounds(north=35.6762, south=35.6695, east=139.6917, west=139.6842)
        ]
        
        for bounds in test_locations:
            transformer = CoordinateTransformer(
                bounds=bounds,
                image_width=1280,
                image_height=1280,
                resolution_mpp=0.1
            )
            
            # Test center point conversion
            center_lat = (bounds.north + bounds.south) / 2
            center_lon = (bounds.east + bounds.west) / 2
            
            x, y = transformer.geographic_to_pixel(center_lat, center_lon)
            lat_back, lon_back = transformer.pixel_to_geographic(x, y)
            
            # Should be accurate regardless of location
            assert abs(lat_back - center_lat) < 0.001
            assert abs(lon_back - center_lon) < 0.001

    @pytest.mark.integration
    def test_large_scale_coordinate_processing(self):
        """Test coordinate transformation performance with large datasets."""
        bounds = GeographicBounds(
            north=-27.4698, south=-27.4705, east=153.0258, west=153.0251
        )
        
        transformer = CoordinateTransformer(
            bounds=bounds,
            image_width=1280,
            image_height=1280,
            resolution_mpp=0.1
        )
        
        # Generate large polyline (simulating complex lane detection)
        num_points = 1000
        large_polyline = []
        
        for i in range(num_points):
            x = 100 + (i * 1080) / num_points  # Spread across image width
            y = 640 + 50 * math.sin(i * 0.01)  # Sinusoidal pattern
            large_polyline.append([x, y])
        
        analyzer = GeographicAnalyzer(transformer)
        
        # Should handle large polylines efficiently
        analysis = analyzer.analyze_infrastructure_geometry(large_polyline)
        
        assert analysis['length_meters'] > 0
        assert analysis['complexity_score'] > 0
        assert len(large_polyline) >= num_points