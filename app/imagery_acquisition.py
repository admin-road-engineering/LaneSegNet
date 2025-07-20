"""
Aerial imagery acquisition from geographic coordinates.
Supports multiple data sources for comprehensive coverage.
"""

import logging
import requests
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional, Dict, Any
import os
from .schemas import GeographicBounds

logger = logging.getLogger(__name__)

class ImageryProvider:
    """Base class for aerial imagery providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fetch aerial imagery for the given geographic bounds.
        
        Args:
            bounds: Geographic bounding box
            resolution_mpp: Desired meters per pixel resolution
            
        Returns:
            Tuple of (image_array, metadata)
        """
        raise NotImplementedError

class GoogleEarthEngineProvider(ImageryProvider):
    """Google Earth Engine imagery provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Initialize Google Earth Engine connection
        # Note: Requires proper authentication setup
        
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fetch high-resolution satellite imagery from Google Earth Engine."""
        try:
            # Calculate required zoom level from resolution
            zoom_level = self._resolution_to_zoom(resolution_mpp)
            
            # Google Earth Engine API call
            # This is a simplified example - actual implementation requires proper GEE setup
            image_url = self._build_gee_url(bounds, zoom_level)
            
            response = requests.get(image_url, headers={'Authorization': f'Bearer {self.api_key}'})
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            image_array = np.array(image)
            
            metadata = {
                'source': 'Google Earth Engine',
                'resolution_mpp': resolution_mpp,
                'zoom_level': zoom_level,
                'bounds': bounds,
                'acquisition_date': 'recent'  # GEE provides recent composite
            }
            
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch Google Earth Engine imagery: {e}")
            raise
    
    def _resolution_to_zoom(self, resolution_mpp: float) -> int:
        """Convert meters per pixel to appropriate zoom level."""
        # Rough conversion - adjust based on actual requirements
        if resolution_mpp <= 0.1:
            return 20  # Very high resolution
        elif resolution_mpp <= 0.5:
            return 18
        elif resolution_mpp <= 1.0:
            return 16
        else:
            return 14
    
    def _build_gee_url(self, bounds: GeographicBounds, zoom: int) -> str:
        """Build Google Earth Engine API URL."""
        # Simplified URL building - actual GEE implementation more complex
        return f"https://earthengine.googleapis.com/v1alpha/projects/YOUR_PROJECT/maps/tiles/{zoom}/{bounds.west}/{bounds.north}"

class MapboxProvider(ImageryProvider):
    """Mapbox satellite imagery provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
    
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fetch satellite imagery from Mapbox."""
        try:
            # Calculate image dimensions based on bounds and resolution
            width, height = self._calculate_dimensions(bounds, resolution_mpp)
            
            # Build Mapbox Static API URL
            bbox = f"{bounds.west},{bounds.south},{bounds.east},{bounds.north}"
            url = f"{self.base_url}/[{bbox}]/{width}x{height}@2x"
            
            params = {
                'access_token': self.api_key,
                'attribution': 'false',
                'logo': 'false'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            image_array = np.array(image)
            
            metadata = {
                'source': 'Mapbox Satellite',
                'resolution_mpp': resolution_mpp,
                'bounds': bounds,
                'image_dimensions': (width, height),
                'acquisition_date': 'recent'
            }
            
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch Mapbox imagery: {e}")
            raise
    
    def _calculate_dimensions(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[int, int]:
        """Calculate required image dimensions."""
        # Rough calculation - should use proper geographic projection
        lat_diff = bounds.north - bounds.south
        lon_diff = bounds.east - bounds.west
        
        # Convert to meters (approximate)
        lat_meters = lat_diff * 111000  # degrees to meters
        lon_meters = lon_diff * 111000 * np.cos(np.radians((bounds.north + bounds.south) / 2))
        
        width = int(lon_meters / resolution_mpp)
        height = int(lat_meters / resolution_mpp)
        
        # Mapbox limits
        width = min(width, 1280)
        height = min(height, 1280)
        
        return width, height

class LocalImageryProvider(ImageryProvider):
    """Provider for local imagery files (for testing/development)."""
    
    def __init__(self, imagery_directory: str):
        super().__init__()
        self.imagery_directory = imagery_directory
    
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load imagery from local files."""
        try:
            # For development - load a sample image
            # In practice, would need proper geographic indexing
            image_path = os.path.join(self.imagery_directory, "sample_aerial.jpg")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local imagery not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            metadata = {
                'source': 'Local File',
                'resolution_mpp': resolution_mpp,
                'bounds': bounds,
                'file_path': image_path
            }
            
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to load local imagery: {e}")
            raise

class ImageryAcquisitionManager:
    """Manages multiple imagery providers with fallback capability."""
    
    def __init__(self):
        self.providers = {}
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available imagery providers."""
        # Google Earth Engine
        gee_key = os.getenv('GOOGLE_EARTH_ENGINE_API_KEY')
        if gee_key:
            self.providers['gee'] = GoogleEarthEngineProvider(gee_key)
        
        # Mapbox
        mapbox_key = os.getenv('MAPBOX_API_KEY')
        if mapbox_key:
            self.providers['mapbox'] = MapboxProvider(mapbox_key)
        
        # Local imagery (for development)
        local_dir = os.getenv('LOCAL_IMAGERY_DIR', 'data/imagery')
        if os.path.exists(local_dir):
            self.providers['local'] = LocalImageryProvider(local_dir)
        
        logger.info(f"Initialized imagery providers: {list(self.providers.keys())}")
    
    def fetch_best_imagery(self, bounds: GeographicBounds, resolution_mpp: float, 
                          preferred_provider: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fetch imagery using the best available provider.
        
        Args:
            bounds: Geographic bounding box
            resolution_mpp: Desired resolution in meters per pixel
            preferred_provider: Preferred provider name ('gee', 'mapbox', 'local')
            
        Returns:
            Tuple of (image_array, metadata)
        """
        # Try preferred provider first
        if preferred_provider and preferred_provider in self.providers:
            try:
                return self.providers[preferred_provider].fetch_imagery(bounds, resolution_mpp)
            except Exception as e:
                logger.warning(f"Preferred provider {preferred_provider} failed: {e}")
        
        # Fallback order: GEE -> Mapbox -> Local
        provider_order = ['gee', 'mapbox', 'local']
        
        for provider_name in provider_order:
            if provider_name in self.providers:
                try:
                    logger.info(f"Attempting imagery fetch with {provider_name}")
                    return self.providers[provider_name].fetch_imagery(bounds, resolution_mpp)
                except Exception as e:
                    logger.warning(f"Provider {provider_name} failed: {e}")
                    continue
        
        raise RuntimeError("All imagery providers failed")

# Global imagery manager instance
imagery_manager = ImageryAcquisitionManager()