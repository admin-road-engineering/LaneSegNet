import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import requests
from PIL import Image
import io

from app.imagery_acquisition import (
    ImageryProvider, GoogleEarthEngineProvider, MapboxProvider, 
    EsriSatelliteProvider, LocalImageryProvider, ImageryManager
)
from app.schemas import GeographicBounds
from tests.conftest import assert_valid_coordinates


class TestImageryProvider:
    """Test suite for the base ImageryProvider class."""

    @pytest.mark.unit
    def test_imagery_provider_init(self):
        """Test ImageryProvider initialization."""
        provider = ImageryProvider("test_api_key")
        assert provider.api_key == "test_api_key"
        
        provider_no_key = ImageryProvider()
        assert provider_no_key.api_key is None

    @pytest.mark.unit
    def test_imagery_provider_fetch_not_implemented(self):
        """Test that base class fetch_imagery raises NotImplementedError."""
        provider = ImageryProvider()
        bounds = GeographicBounds(north=1.0, south=0.0, east=1.0, west=0.0)
        
        with pytest.raises(NotImplementedError):
            provider.fetch_imagery(bounds, 0.1)


class TestGoogleEarthEngineProvider:
    """Test suite for GoogleEarthEngineProvider."""

    @pytest.fixture
    def gee_provider(self):
        """Google Earth Engine provider fixture."""
        return GoogleEarthEngineProvider("test_gee_api_key")

    @pytest.mark.unit
    def test_gee_provider_init(self, gee_provider):
        """Test GEE provider initialization."""
        assert gee_provider.api_key == "test_gee_api_key"

    @pytest.mark.unit
    def test_gee_resolution_to_zoom(self, gee_provider):
        """Test resolution to zoom level conversion."""
        assert gee_provider._resolution_to_zoom(0.05) == 20
        assert gee_provider._resolution_to_zoom(0.3) == 18
        assert gee_provider._resolution_to_zoom(0.8) == 16
        assert gee_provider._resolution_to_zoom(1.5) == 14

    @pytest.mark.unit
    def test_gee_build_url(self, gee_provider):
        """Test GEE URL building."""
        bounds = GeographicBounds(north=1.0, south=0.0, east=1.0, west=0.0)
        url = gee_provider._build_gee_url(bounds, 18)
        
        assert "earthengine.googleapis.com" in url
        assert "18" in url
        assert str(bounds.west) in url
        assert str(bounds.north) in url

    @pytest.mark.unit
    def test_gee_fetch_imagery_success(self, gee_provider, test_coordinates):
        """Test successful GEE imagery fetch."""
        # Create mock image
        mock_image = Image.new('RGB', (512, 512), color='green')
        img_bytes = io.BytesIO()
        mock_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Mock requests response
        mock_response = Mock()
        mock_response.content = img_bytes.read()
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            image_array, metadata = gee_provider.fetch_imagery(test_coordinates, 0.1)
        
        # Validate results
        assert isinstance(image_array, np.ndarray)
        assert image_array.shape == (512, 512, 3)
        assert metadata['source'] == 'Google Earth Engine'
        assert metadata['resolution_mpp'] == 0.1
        assert 'zoom_level' in metadata

    @pytest.mark.unit
    def test_gee_fetch_imagery_failure(self, gee_provider, test_coordinates):
        """Test GEE imagery fetch failure handling."""
        # Mock failed request
        with patch('requests.get', side_effect=requests.RequestException("API Error")):
            with pytest.raises(requests.RequestException):
                gee_provider.fetch_imagery(test_coordinates, 0.1)


class TestMapboxProvider:
    """Test suite for MapboxProvider."""

    @pytest.fixture
    def mapbox_provider(self):
        """Mapbox provider fixture."""
        return MapboxProvider("test_mapbox_api_key")

    @pytest.mark.unit
    def test_mapbox_provider_init(self, mapbox_provider):
        """Test Mapbox provider initialization."""
        assert mapbox_provider.api_key == "test_mapbox_api_key"
        assert "mapbox.com" in mapbox_provider.base_url

    @pytest.mark.unit
    def test_mapbox_calculate_dimensions(self, mapbox_provider):
        """Test dimension calculation for Mapbox requests."""
        bounds = GeographicBounds(north=-27.4698, south=-27.4705, east=153.0258, west=153.0251)
        width, height = mapbox_provider._calculate_dimensions(bounds, 0.1)
        
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width > 0 and height > 0
        assert width <= 1280 and height <= 1280  # Mapbox limits

    @pytest.mark.unit
    def test_mapbox_fetch_imagery_success(self, mapbox_provider, test_coordinates):
        """Test successful Mapbox imagery fetch."""
        # Create mock image
        mock_image = Image.new('RGB', (512, 512), color='blue')
        img_bytes = io.BytesIO()
        mock_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Mock requests response
        mock_response = Mock()
        mock_response.content = img_bytes.read()
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            image_array, metadata = mapbox_provider.fetch_imagery(test_coordinates, 0.1)
        
        # Validate results
        assert isinstance(image_array, np.ndarray)
        assert image_array.shape == (512, 512, 3)
        assert metadata['source'] == 'Mapbox Satellite'
        assert metadata['resolution_mpp'] == 0.1
        assert 'image_dimensions' in metadata

    @pytest.mark.unit
    def test_mapbox_fetch_imagery_failure(self, mapbox_provider, test_coordinates):
        """Test Mapbox imagery fetch failure handling."""
        with patch('requests.get', side_effect=requests.RequestException("API Error")):
            with pytest.raises(requests.RequestException):
                mapbox_provider.fetch_imagery(test_coordinates, 0.1)


class TestEsriSatelliteProvider:
    """Test suite for EsriSatelliteProvider."""

    @pytest.fixture
    def esri_provider(self):
        """Esri provider fixture."""
        return EsriSatelliteProvider()

    @pytest.mark.unit
    def test_esri_provider_init(self, esri_provider):
        """Test Esri provider initialization."""
        assert esri_provider.api_key is None  # No API key required
        assert "arcgisonline.com" in esri_provider.tile_server

    @pytest.mark.unit
    def test_esri_resolution_to_zoom(self, esri_provider):
        """Test Esri resolution to zoom conversion."""
        bounds = GeographicBounds(north=-27.4698, south=-27.4705, east=153.0258, west=153.0251)
        
        zoom_high = esri_provider._resolution_to_zoom(0.2, bounds)
        zoom_low = esri_provider._resolution_to_zoom(2.0, bounds)
        
        assert zoom_high > zoom_low  # Higher resolution should give higher zoom
        assert 14 <= zoom_low <= 19
        assert 14 <= zoom_high <= 19

    @pytest.mark.unit
    def test_esri_deg2num_conversion(self, esri_provider):
        """Test lat/lon to tile number conversion."""
        # Test known conversion (Brisbane area)
        lat, lon = -27.4701, 153.0254
        zoom = 16
        
        x_tile, y_tile = esri_provider._deg2num(lat, lon, zoom)
        
        assert isinstance(x_tile, int)
        assert isinstance(y_tile, int)
        assert x_tile >= 0 and y_tile >= 0

    @pytest.mark.unit
    def test_esri_num2deg_conversion(self, esri_provider):
        """Test tile number to lat/lon conversion."""
        # Test round-trip conversion
        original_lat, original_lon = -27.4701, 153.0254
        zoom = 16
        
        x_tile, y_tile = esri_provider._deg2num(original_lat, original_lon, zoom)
        converted_lat, converted_lon = esri_provider._num2deg(x_tile, y_tile, zoom)
        
        # Should be close to original (within tile precision)
        assert abs(converted_lat - original_lat) < 0.01
        assert abs(converted_lon - original_lon) < 0.01

    @pytest.mark.unit
    def test_esri_get_tile_bounds(self, esri_provider, test_coordinates):
        """Test tile bounds calculation."""
        zoom = 16
        tile_bounds = esri_provider._get_tile_bounds(test_coordinates, zoom)
        
        assert 'x_min' in tile_bounds
        assert 'x_max' in tile_bounds
        assert 'y_min' in tile_bounds
        assert 'y_max' in tile_bounds
        assert 'tiles_x' in tile_bounds
        assert 'tiles_y' in tile_bounds
        assert tile_bounds['zoom'] == zoom
        
        # Bounds should be sensible
        assert tile_bounds['x_max'] >= tile_bounds['x_min']
        assert tile_bounds['y_max'] >= tile_bounds['y_min']
        assert tile_bounds['tiles_x'] > 0
        assert tile_bounds['tiles_y'] > 0

    @pytest.mark.unit
    def test_esri_fetch_imagery_mock(self, esri_provider, test_coordinates):
        """Test Esri imagery fetch with mocked dependencies."""
        # Mock the tile downloading and stitching methods
        mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        with patch.object(esri_provider, '_download_and_stitch_tiles', return_value=mock_image), \
             patch.object(esri_provider, '_crop_to_bounds', return_value=mock_image):
            
            image_array, metadata = esri_provider.fetch_imagery(test_coordinates, 0.1)
        
        # Validate results
        assert isinstance(image_array, np.ndarray)
        assert image_array.shape == (512, 512, 3)
        assert metadata['source'] == 'Esri World Imagery'
        assert metadata['resolution_mpp'] == 0.1
        assert 'zoom_level' in metadata
        assert 'tiles_used' in metadata


class TestLocalImageryProvider:
    """Test suite for LocalImageryProvider."""

    @pytest.fixture
    def local_provider(self):
        """Local imagery provider fixture."""
        return LocalImageryProvider(imagery_dir="data/imgs")

    @pytest.mark.unit
    def test_local_provider_init(self, local_provider):
        """Test local provider initialization."""
        assert local_provider.imagery_dir == "data/imgs"

    @pytest.mark.unit
    def test_local_provider_list_images(self, local_provider):
        """Test listing available local images."""
        # Mock os.listdir to return test images
        with patch('os.listdir', return_value=['00000001.jpg', '00000002.jpg', 'not_image.txt']), \
             patch('os.path.exists', return_value=True):
            
            images = local_provider._list_available_images()
            
            # Should filter to only image files
            assert len(images) == 2
            assert '00000001.jpg' in images
            assert '00000002.jpg' in images
            assert 'not_image.txt' not in images

    @pytest.mark.unit 
    def test_local_provider_fetch_imagery_success(self, local_provider, test_coordinates):
        """Test successful local imagery fetch."""
        # Mock image loading
        mock_image = Image.new('RGB', (1280, 1280), color='red')
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['00000001.jpg']), \
             patch('PIL.Image.open', return_value=mock_image):
            
            image_array, metadata = local_provider.fetch_imagery(test_coordinates, 0.1)
        
        # Validate results
        assert isinstance(image_array, np.ndarray)
        assert image_array.shape == (1280, 1280, 3)
        assert metadata['source'] == 'local_imagery'
        assert metadata['resolution_mpp'] == 0.1
        assert 'selected_image' in metadata

    @pytest.mark.unit
    def test_local_provider_no_images_available(self, local_provider, test_coordinates):
        """Test behavior when no local images available."""
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=[]):
            
            with pytest.raises(FileNotFoundError, match="No local aerial images"):
                local_provider.fetch_imagery(test_coordinates, 0.1)

    @pytest.mark.unit
    def test_local_provider_directory_not_found(self, local_provider, test_coordinates):
        """Test behavior when imagery directory doesn't exist."""
        with patch('os.path.exists', return_value=False):
            
            with pytest.raises(FileNotFoundError, match="Local imagery directory"):
                local_provider.fetch_imagery(test_coordinates, 0.1)


class TestImageryManager:
    """Test suite for ImageryManager."""

    @pytest.fixture
    def imagery_manager(self):
        """Imagery manager fixture."""
        return ImageryManager()

    @pytest.mark.unit
    def test_imagery_manager_init(self, imagery_manager):
        """Test imagery manager initialization."""
        assert 'local' in imagery_manager.providers
        assert 'esri' in imagery_manager.providers
        # Note: GEE and Mapbox providers only added if API keys available

    @pytest.mark.unit
    def test_imagery_manager_add_provider(self, imagery_manager):
        """Test adding custom provider."""
        custom_provider = Mock(spec=ImageryProvider)
        imagery_manager.add_provider('custom', custom_provider)
        
        assert 'custom' in imagery_manager.providers
        assert imagery_manager.providers['custom'] == custom_provider

    @pytest.mark.unit
    async def test_imagery_manager_acquire_imagery_success(self, imagery_manager, test_coordinates):
        """Test successful imagery acquisition."""
        # Mock a provider to return test imagery
        mock_provider = Mock()
        mock_provider.fetch_imagery.return_value = (
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            {'source': 'test_provider', 'resolution_mpp': 0.1}
        )
        
        imagery_manager.providers['test'] = mock_provider
        
        result = await imagery_manager.acquire_imagery(
            bounds=test_coordinates,
            resolution_mpp=0.1,
            preferred_provider='test'
        )
        
        assert 'image' in result
        assert 'metadata' in result
        assert isinstance(result['image'], np.ndarray)
        assert result['metadata']['source'] == 'test_provider'

    @pytest.mark.unit
    async def test_imagery_manager_fallback_providers(self, imagery_manager, test_coordinates):
        """Test fallback to other providers when preferred fails."""
        # Mock first provider to fail
        failing_provider = Mock()
        failing_provider.fetch_imagery.side_effect = Exception("Provider failed")
        
        # Mock second provider to succeed
        working_provider = Mock()
        working_provider.fetch_imagery.return_value = (
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            {'source': 'backup_provider', 'resolution_mpp': 0.1}
        )
        
        imagery_manager.providers = {
            'failing': failing_provider,
            'working': working_provider
        }
        
        result = await imagery_manager.acquire_imagery(
            bounds=test_coordinates,
            resolution_mpp=0.1,
            preferred_provider='failing'
        )
        
        # Should fallback to working provider
        assert result['metadata']['source'] == 'backup_provider'

    @pytest.mark.unit
    async def test_imagery_manager_all_providers_fail(self, imagery_manager, test_coordinates):
        """Test behavior when all providers fail."""
        # Mock all providers to fail
        failing_provider1 = Mock()
        failing_provider1.fetch_imagery.side_effect = Exception("Provider 1 failed")
        
        failing_provider2 = Mock()
        failing_provider2.fetch_imagery.side_effect = Exception("Provider 2 failed")
        
        imagery_manager.providers = {
            'fail1': failing_provider1,
            'fail2': failing_provider2
        }
        
        with pytest.raises(RuntimeError, match="All imagery providers failed"):
            await imagery_manager.acquire_imagery(
                bounds=test_coordinates,
                resolution_mpp=0.1
            )

    @pytest.mark.unit
    async def test_imagery_manager_invalid_preferred_provider(self, imagery_manager, test_coordinates):
        """Test handling of invalid preferred provider."""
        # Mock a working provider
        working_provider = Mock()
        working_provider.fetch_imagery.return_value = (
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            {'source': 'working_provider', 'resolution_mpp': 0.1}
        )
        
        imagery_manager.providers = {'working': working_provider}
        
        # Request non-existent provider - should fallback to available
        result = await imagery_manager.acquire_imagery(
            bounds=test_coordinates,
            resolution_mpp=0.1,
            preferred_provider='nonexistent'
        )
        
        assert result['metadata']['source'] == 'working_provider'

    @pytest.mark.unit
    def test_imagery_manager_provider_selection_priority(self, imagery_manager):
        """Test provider selection follows priority order."""
        # Test that local provider is prioritized when available
        assert 'local' in imagery_manager.providers
        
        # If we have a local provider, it should be tried first
        provider_list = list(imagery_manager.providers.keys())
        assert 'local' in provider_list[:2]  # Should be in top priority


class TestImageryAcquisitionIntegration:
    """Integration tests for imagery acquisition workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_full_imagery_acquisition_workflow(self, test_coordinates):
        """Test complete imagery acquisition workflow."""
        # This would test actual API calls in a real environment
        # For now, we'll mock the external dependencies
        
        manager = ImageryManager()
        
        # Mock local provider with real-looking data
        mock_provider = Mock()
        mock_provider.fetch_imagery.return_value = (
            np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8),
            {
                'source': 'local_imagery',
                'resolution_mpp': 0.1,
                'acquisition_date': '2024-01-15',
                'selected_image': '00000001.jpg'
            }
        )
        
        manager.providers['local'] = mock_provider
        
        result = await manager.acquire_imagery(
            bounds=test_coordinates,
            resolution_mpp=0.1,
            preferred_provider='local'
        )
        
        # Validate complete workflow result
        assert 'image' in result
        assert 'metadata' in result
        
        image = result['image']
        metadata = result['metadata']
        
        # Validate image properties
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Height, width, channels
        assert image.shape[2] == 3  # RGB
        assert image.dtype == np.uint8
        
        # Validate metadata completeness
        required_metadata = ['source', 'resolution_mpp']
        for field in required_metadata:
            assert field in metadata
        
        # Validate coordinate consistency
        assert_valid_coordinates(test_coordinates)