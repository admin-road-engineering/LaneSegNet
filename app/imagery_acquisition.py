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

class OpenStreetMapProvider(ImageryProvider):
    """OpenStreetMap tile server imagery provider (free, no API key required)."""
    
    def __init__(self, tile_server: str = "https://tile.openstreetmap.org"):
        super().__init__()
        self.tile_server = tile_server
        # Alternative OSM tile servers for better coverage:
        # https://a.tile.openstreetmap.org
        # https://b.tile.openstreetmap.org  
        # https://c.tile.openstreetmap.org
    
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fetch satellite/aerial imagery from OpenStreetMap tile servers."""
        try:
            # Calculate appropriate zoom level for the requested resolution
            zoom_level = self._resolution_to_zoom(resolution_mpp, bounds)
            
            # Get tile bounds for the geographic area
            tile_bounds = self._get_tile_bounds(bounds, zoom_level)
            
            # Download and stitch tiles together
            image_array = self._download_and_stitch_tiles(tile_bounds, zoom_level)
            
            # Crop to exact bounds if needed
            cropped_image = self._crop_to_bounds(image_array, bounds, tile_bounds)
            
            metadata = {
                'source': 'OpenStreetMap',
                'resolution_mpp': resolution_mpp,
                'zoom_level': zoom_level,
                'bounds': bounds,
                'tile_server': self.tile_server,
                'tiles_used': f"{tile_bounds['tiles_x']}x{tile_bounds['tiles_y']}",
                'note': 'Free OSM imagery - may not be high-resolution aerial'
            }
            
            return cropped_image, metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch OpenStreetMap imagery: {e}")
            raise
    
    def _resolution_to_zoom(self, resolution_mpp: float, bounds: GeographicBounds) -> int:
        """Calculate appropriate OSM zoom level for requested resolution."""
        # OSM zoom levels: higher zoom = more detail
        # At equator: zoom 18 ≈ 0.6m/pixel, zoom 16 ≈ 2.4m/pixel
        lat_avg = (bounds.north + bounds.south) / 2
        
        # Adjust for latitude (tiles get smaller away from equator)
        cos_lat = np.cos(np.radians(lat_avg))
        
        if resolution_mpp <= 0.6 * cos_lat:
            return 18  # Very high detail
        elif resolution_mpp <= 1.2 * cos_lat:
            return 17
        elif resolution_mpp <= 2.4 * cos_lat:
            return 16
        elif resolution_mpp <= 4.8 * cos_lat:
            return 15
        else:
            return 14  # Lower detail for large areas
    
    def _deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to OSM tile numbers."""
        lat_rad = np.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def _num2deg(self, xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert OSM tile numbers to lat/lon."""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
        lat_deg = np.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def _get_tile_bounds(self, bounds: GeographicBounds, zoom: int) -> Dict:
        """Calculate which OSM tiles cover the requested area."""
        # Get tile coordinates for corners
        x_min, y_max = self._deg2num(bounds.south, bounds.west, zoom)  # Note: y is flipped
        x_max, y_min = self._deg2num(bounds.north, bounds.east, zoom)
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'tiles_x': x_max - x_min + 1,
            'tiles_y': y_max - y_min + 1,
            'zoom': zoom
        }
    
    def _download_and_stitch_tiles(self, tile_bounds: Dict, zoom: int) -> np.ndarray:
        """Download OSM tiles and stitch them together."""
        import requests
        from PIL import Image
        
        tiles_x = tile_bounds['tiles_x']
        tiles_y = tile_bounds['tiles_y']
        
        # OSM tiles are 256x256 pixels
        tile_size = 256
        full_width = tiles_x * tile_size
        full_height = tiles_y * tile_size
        
        # Create output image
        full_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)
        
        # Download each tile
        for dy in range(tiles_y):
            for dx in range(tiles_x):
                x_tile = tile_bounds['x_min'] + dx
                y_tile = tile_bounds['y_min'] + dy
                
                try:
                    # OSM tile URL format
                    tile_url = f"{self.tile_server}/{zoom}/{x_tile}/{y_tile}.png"
                    
                    # Download tile with proper headers (OSM requires User-Agent)
                    headers = {
                        'User-Agent': 'LaneSegNet/1.0 (Infrastructure Analysis)'
                    }
                    response = requests.get(tile_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    # Load tile image
                    tile_img = Image.open(io.BytesIO(response.content)).convert('RGB')
                    tile_array = np.array(tile_img)
                    
                    # Place in full image
                    y_start = dy * tile_size
                    y_end = y_start + tile_size
                    x_start = dx * tile_size  
                    x_end = x_start + tile_size
                    
                    full_image[y_start:y_end, x_start:x_end] = tile_array
                    
                except Exception as e:
                    logger.warning(f"Failed to download tile {x_tile},{y_tile}: {e}")
                    # Leave as black/empty - continue with other tiles
        
        return full_image
    
    def _crop_to_bounds(self, image_array: np.ndarray, target_bounds: GeographicBounds, tile_bounds: Dict) -> np.ndarray:
        """Crop the stitched image to exact geographic bounds."""
        # For simplicity, return the full stitched image
        # In production, this would do precise cropping to exact bounds
        return image_array

class LocalImageryProvider(ImageryProvider):
    """Provider for local imagery files (for testing/development)."""
    
    def __init__(self, imagery_directory: str):
        super().__init__()
        self.imagery_directory = imagery_directory
    
    def fetch_imagery(self, bounds: GeographicBounds, resolution_mpp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load imagery from local files."""
        try:
            # For development - randomly select from available images
            # In practice, would need proper geographic indexing
            available_images = [f for f in os.listdir(self.imagery_directory) if f.endswith('.jpg')]
            
            if not available_images:
                raise FileNotFoundError(f"No imagery found in directory: {self.imagery_directory}")
            
            # Select a random image for development
            import random
            selected_image = random.choice(available_images)
            image_path = os.path.join(self.imagery_directory, selected_image)
            
            logger.info(f"Loading local development image: {selected_image}")
            
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            metadata = {
                'source': 'Local Development File',
                'resolution_mpp': resolution_mpp,
                'bounds': bounds,
                'file_path': image_path,
                'selected_image': selected_image,
                'note': 'Development mode - using random local image'
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
        # OpenStreetMap (Free, no API key required)
        self.providers['osm'] = OpenStreetMapProvider()
        
        # Google Earth Engine
        gee_key = os.getenv('GOOGLE_EARTH_ENGINE_API_KEY')
        if gee_key:
            self.providers['gee'] = GoogleEarthEngineProvider(gee_key)
        
        # Mapbox
        mapbox_key = os.getenv('MAPBOX_API_KEY')
        if mapbox_key:
            self.providers['mapbox'] = MapboxProvider(mapbox_key)
        
        # Local imagery (for development) - use existing data/imgs directory
        local_dir = os.getenv('LOCAL_IMAGERY_DIR', 'data/imgs')
        if os.path.exists(local_dir):
            self.providers['local'] = LocalImageryProvider(local_dir)
        
        logger.info(f"Initialized imagery providers: {list(self.providers.keys())}")
    
    async def acquire_imagery(self, bounds: GeographicBounds, resolution_mpp: float, 
                             preferred_provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Acquire imagery using the best available provider (async wrapper).
        
        Args:
            bounds: Geographic bounding box
            resolution_mpp: Desired resolution in meters per pixel
            preferred_provider: Preferred provider name ('gee', 'mapbox', 'local')
            
        Returns:
            Dictionary with 'image' and 'metadata' keys
        """
        image_array, metadata = self.fetch_best_imagery(bounds, resolution_mpp, preferred_provider)
        return {
            "image": image_array,
            "metadata": metadata
        }
    
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
        
        # Fallback order: OSM -> GEE -> Mapbox -> Local
        provider_order = ['osm', 'gee', 'mapbox', 'local']
        
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