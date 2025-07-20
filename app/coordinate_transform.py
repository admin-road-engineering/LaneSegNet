"""
Coordinate transformation utilities for converting between 
geographic coordinates (lat/lon) and image pixel coordinates.
"""

import logging
import numpy as np
import math
from typing import List, Tuple, Dict
from .schemas import GeographicBounds, GeographicPoint

logger = logging.getLogger(__name__)

class CoordinateTransformer:
    """Handles coordinate transformations between geographic and pixel coordinates."""
    
    def __init__(self, bounds: GeographicBounds, image_width: int, image_height: int, resolution_mpp: float):
        """
        Initialize coordinate transformer.
        
        Args:
            bounds: Geographic bounds of the image
            image_width: Image width in pixels
            image_height: Image height in pixels
            resolution_mpp: Meters per pixel resolution
        """
        self.bounds = bounds
        self.image_width = image_width
        self.image_height = image_height
        self.resolution_mpp = resolution_mpp
        
        # Calculate transformation parameters
        self._calculate_transform_params()
    
    def _calculate_transform_params(self):
        """Calculate transformation parameters."""
        # Geographic extents
        self.lat_extent = self.bounds.north - self.bounds.south
        self.lon_extent = self.bounds.east - self.bounds.west
        
        # Pixels per degree
        self.pixels_per_lat_degree = self.image_height / self.lat_extent
        self.pixels_per_lon_degree = self.image_width / self.lon_extent
        
        # Meters per degree (approximate)
        center_lat = (self.bounds.north + self.bounds.south) / 2
        self.meters_per_lat_degree = 111000  # Constant
        self.meters_per_lon_degree = 111000 * math.cos(math.radians(center_lat))
        
        logger.debug(f"Transform params: {self.pixels_per_lat_degree:.2f} px/lat°, {self.pixels_per_lon_degree:.2f} px/lon°")
    
    def geographic_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        # Calculate relative position within bounds
        lat_ratio = (lat - self.bounds.south) / self.lat_extent
        lon_ratio = (lon - self.bounds.west) / self.lon_extent
        
        # Convert to pixel coordinates
        # Note: Image coordinates typically have (0,0) at top-left
        x = int(lon_ratio * self.image_width)
        y = int((1 - lat_ratio) * self.image_height)  # Flip Y axis
        
        # Clamp to image bounds
        x = max(0, min(x, self.image_width - 1))
        y = max(0, min(y, self.image_height - 1))
        
        return x, y
    
    def pixel_to_geographic(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            
        Returns:
            Tuple of (lat, lon) geographic coordinates
        """
        # Calculate relative position within image
        lon_ratio = x / self.image_width
        lat_ratio = 1 - (y / self.image_height)  # Flip Y axis
        
        # Convert to geographic coordinates
        lat = self.bounds.south + (lat_ratio * self.lat_extent)
        lon = self.bounds.west + (lon_ratio * self.lon_extent)
        
        return lat, lon
    
    def pixel_distance_to_meters(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to meters.
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            Distance in meters
        """
        return pixel_distance * self.resolution_mpp
    
    def meters_to_pixel_distance(self, meters: float) -> float:
        """
        Convert meters to pixel distance.
        
        Args:
            meters: Distance in meters
            
        Returns:
            Distance in pixels
        """
        return meters / self.resolution_mpp
    
    def calculate_area_sqm(self, pixel_area: float) -> float:
        """
        Convert pixel area to square meters.
        
        Args:
            pixel_area: Area in pixels
            
        Returns:
            Area in square meters
        """
        return pixel_area * (self.resolution_mpp ** 2)
    
    def transform_polyline_to_geographic(self, pixel_points: List[List[float]]) -> List[GeographicPoint]:
        """
        Transform a polyline from pixel coordinates to geographic coordinates.
        
        Args:
            pixel_points: List of [x, y] pixel coordinates
            
        Returns:
            List of GeographicPoint objects
        """
        geographic_points = []
        
        for point in pixel_points:
            x, y = point[0], point[1]
            lat, lon = self.pixel_to_geographic(int(x), int(y))
            geographic_points.append(GeographicPoint(latitude=lat, longitude=lon))
        
        return geographic_points
    
    def calculate_polyline_length_meters(self, pixel_points: List[List[float]]) -> float:
        """
        Calculate the length of a polyline in meters.
        
        Args:
            pixel_points: List of [x, y] pixel coordinates
            
        Returns:
            Length in meters
        """
        if len(pixel_points) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(1, len(pixel_points)):
            # Calculate pixel distance
            dx = pixel_points[i][0] - pixel_points[i-1][0]
            dy = pixel_points[i][1] - pixel_points[i-1][1]
            pixel_distance = math.sqrt(dx**2 + dy**2)
            
            # Convert to meters
            meters = self.pixel_distance_to_meters(pixel_distance)
            total_length += meters
        
        return total_length
    
    def calculate_geographic_distance(self, point1: GeographicPoint, point2: GeographicPoint) -> float:
        """
        Calculate distance between two geographic points using Haversine formula.
        
        Args:
            point1: First geographic point
            point2: Second geographic point
            
        Returns:
            Distance in meters
        """
        # Haversine formula for great circle distance
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(point1.latitude)
        lat2_rad = math.radians(point2.latitude)
        dlat_rad = math.radians(point2.latitude - point1.latitude)
        dlon_rad = math.radians(point2.longitude - point1.longitude)
        
        a = (math.sin(dlat_rad/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate if coordinates are within the image bounds.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are within bounds
        """
        return (self.bounds.south <= lat <= self.bounds.north and 
                self.bounds.west <= lon <= self.bounds.east)

class GeographicAnalyzer:
    """Provides geographic analysis capabilities for road infrastructure."""
    
    def __init__(self, transformer: CoordinateTransformer):
        self.transformer = transformer
    
    def analyze_road_width(self, centerline_points: List[List[float]], 
                          left_edge_points: List[List[float]], 
                          right_edge_points: List[List[float]]) -> List[float]:
        """
        Calculate road width at multiple points along centerline.
        
        Args:
            centerline_points: Center line pixel coordinates
            left_edge_points: Left edge pixel coordinates  
            right_edge_points: Right edge pixel coordinates
            
        Returns:
            List of widths in meters at each centerline point
        """
        widths = []
        
        for center_point in centerline_points:
            # Find closest points on each edge
            left_dist, left_point = self._find_closest_point(center_point, left_edge_points)
            right_dist, right_point = self._find_closest_point(center_point, right_edge_points)
            
            # Calculate total width
            left_meters = self.transformer.pixel_distance_to_meters(left_dist)
            right_meters = self.transformer.pixel_distance_to_meters(right_dist)
            total_width = left_meters + right_meters
            
            widths.append(total_width)
        
        return widths
    
    def _find_closest_point(self, target_point: List[float], 
                           candidate_points: List[List[float]]) -> Tuple[float, List[float]]:
        """Find the closest point from candidates to target."""
        min_distance = float('inf')
        closest_point = None
        
        for candidate in candidate_points:
            dx = candidate[0] - target_point[0]
            dy = candidate[1] - target_point[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = candidate
        
        return min_distance, closest_point
    
    def calculate_lane_spacing(self, lane_markings: List[List[List[float]]]) -> Dict[str, float]:
        """
        Calculate spacing between parallel lane markings.
        
        Args:
            lane_markings: List of lane marking polylines
            
        Returns:
            Dictionary with spacing statistics
        """
        if len(lane_markings) < 2:
            return {"average_spacing_m": 0.0, "min_spacing_m": 0.0, "max_spacing_m": 0.0}
        
        spacings = []
        
        # Calculate spacing between adjacent lanes
        for i in range(len(lane_markings) - 1):
            lane1 = lane_markings[i]
            lane2 = lane_markings[i + 1]
            
            # Sample spacing at multiple points
            num_samples = min(len(lane1), len(lane2), 10)
            for j in range(0, num_samples):
                idx1 = int(j * len(lane1) / num_samples)
                idx2 = int(j * len(lane2) / num_samples)
                
                point1 = lane1[idx1]
                point2 = lane2[idx2]
                
                # Calculate pixel distance
                dx = point2[0] - point1[0]
                dy = point2[1] - point1[1]
                pixel_distance = math.sqrt(dx**2 + dy**2)
                
                # Convert to meters
                meters = self.transformer.pixel_distance_to_meters(pixel_distance)
                spacings.append(meters)
        
        if not spacings:
            return {"average_spacing_m": 0.0, "min_spacing_m": 0.0, "max_spacing_m": 0.0}
        
        return {
            "average_spacing_m": np.mean(spacings),
            "min_spacing_m": np.min(spacings),
            "max_spacing_m": np.max(spacings)
        }