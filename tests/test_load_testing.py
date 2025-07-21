"""
Load testing for LaneSegNet API endpoints using Locust.
Tests concurrent user scenarios and performance under load.
"""

import json
import random
from io import BytesIO
from PIL import Image
import tempfile

from locust import HttpUser, task, between


class LaneSegNetUser(HttpUser):
    """Simulates a user interacting with the LaneSegNet API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup test data for the user session."""
        # Test coordinates (Brisbane CBD area)
        self.test_coordinates = [
            {
                "north": -27.4698,
                "south": -27.4705,
                "east": 153.0258,
                "west": 153.0251
            },
            {
                "north": -27.4700,
                "south": -27.4707,
                "east": 153.0260,
                "west": 153.0253
            },
            {
                "north": -27.4696,
                "south": -27.4703,
                "east": 153.0256,
                "west": 153.0249
            }
        ]
        
        # Create test image for upload tests
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self):
        """Create a test image for upload testing."""
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task(5)  # Higher weight = more frequent execution
    def test_health_check(self):
        """Test health check endpoint (most frequent)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "status" in data and "model_loaded" in data:
                    response.success()
                else:
                    response.failure("Health check missing required fields")
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(3)
    def test_analyze_road_infrastructure(self):
        """Test coordinate-based road infrastructure analysis."""
        coords = random.choice(self.test_coordinates)
        
        params = {
            "analysis_type": random.choice(["comprehensive", "lane_markings", "pavements"]),
            "resolution": random.choice([0.1, 0.15, 0.2]),
            "model_type": random.choice(["swin", "lanesegnet"]),
            "visualize": False
        }
        
        with self.client.post(
            "/analyze_road_infrastructure",
            json=coords,
            params=params,
            catch_response=True,
            timeout=30  # 30 second timeout for analysis
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = [
                        "infrastructure_elements", 
                        "analysis_summary", 
                        "image_metadata",
                        "processing_time_ms"
                    ]
                    
                    if all(field in data for field in required_fields):
                        # Check processing time is reasonable
                        if data["processing_time_ms"] < 5000:  # Less than 5 seconds
                            response.success()
                        else:
                            response.failure(f"Processing took too long: {data['processing_time_ms']}ms")
                    else:
                        response.failure("Response missing required fields")
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                response.failure("Model not loaded - service unavailable")
            elif response.status_code == 400:
                response.failure("Bad request - invalid coordinates")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def test_analyze_image_upload(self):
        """Test image upload analysis."""
        files = {"image": ("test.jpg", self.test_image_data, "image/jpeg")}
        data = {
            "analysis_type": "comprehensive",
            "resolution": 0.1,
            "model_type": "swin"
        }
        
        with self.client.post(
            "/analyze_image",
            files=files,
            data=data,
            catch_response=True,
            timeout=30
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "infrastructure_elements" in result and "processing_time_ms" in result:
                        response.success()
                    else:
                        response.failure("Image analysis response missing required fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response from image analysis")
            elif response.status_code == 503:
                response.failure("Model not loaded for image analysis")
            elif response.status_code == 400:
                response.failure("Bad request for image analysis")
            else:
                response.failure(f"Image analysis failed with status {response.status_code}")
    
    @task(1)
    def test_visualization_endpoint(self):
        """Test visualization generation (least frequent due to higher load)."""
        coords = random.choice(self.test_coordinates)
        
        params = {
            "resolution": 0.1,
            "viz_type": random.choice(["side_by_side", "overlay", "annotated"]),
            "show_labels": random.choice([True, False]),
            "show_confidence": random.choice([True, False])
        }
        
        with self.client.post(
            "/visualize_infrastructure",
            json=coords,
            params=params,
            catch_response=True,
            timeout=45  # Longer timeout for visualization
        ) as response:
            if response.status_code == 200:
                if response.headers.get("content-type") == "image/jpeg":
                    # Check that we got a reasonable sized image
                    if len(response.content) > 1000:  # At least 1KB
                        response.success()
                    else:
                        response.failure("Visualization image too small")
                else:
                    response.failure("Visualization did not return image/jpeg")
            elif response.status_code == 503:
                response.failure("Model not loaded for visualization")
            else:
                response.failure(f"Visualization failed with status {response.status_code}")
    
    @task(1)
    def test_visualizer_interface(self):
        """Test web visualizer interface."""
        with self.client.get("/visualizer", catch_response=True) as response:
            if response.status_code == 200:
                if "LaneSegNet Visualizer" in response.text:
                    response.success()
                else:
                    response.failure("Visualizer interface missing expected content")
            else:
                response.failure(f"Visualizer interface failed with status {response.status_code}")


class ConcurrentAnalysisUser(HttpUser):
    """Simulates heavy concurrent analysis workloads."""
    
    wait_time = between(0.5, 2)  # Faster requests for load testing
    
    def on_start(self):
        """Setup for concurrent testing."""
        self.coords = {
            "north": -27.4698,
            "south": -27.4705,
            "east": 153.0258,
            "west": 153.0251
        }
    
    @task(10)
    def concurrent_analysis_requests(self):
        """Generate concurrent analysis requests."""
        # Vary parameters to create different workloads
        params = {
            "analysis_type": "comprehensive",
            "resolution": random.uniform(0.1, 0.3),
            "model_type": random.choice(["swin", "lanesegnet"])
        }
        
        # Add small random variations to coordinates
        varied_coords = {
            "north": self.coords["north"] + random.uniform(-0.001, 0.001),
            "south": self.coords["south"] + random.uniform(-0.001, 0.001),
            "east": self.coords["east"] + random.uniform(-0.001, 0.001),
            "west": self.coords["west"] + random.uniform(-0.001, 0.001),
        }
        
        with self.client.post(
            "/analyze_road_infrastructure",
            json=varied_coords,
            params=params,
            catch_response=True,
            timeout=60  # Longer timeout for concurrent load
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Performance requirement: under 1000ms for concurrent requests
                if data.get("processing_time_ms", 0) < 1000:
                    response.success()
                else:
                    response.failure(f"Too slow under load: {data.get('processing_time_ms')}ms")
            else:
                response.failure(f"Failed under concurrent load: {response.status_code}")


class PerformanceTestScenarios:
    """
    Performance test scenarios for different load patterns.
    
    Usage:
        # Run basic load test
        locust -f tests/test_load_testing.py --host=http://localhost:8010
        
        # Run specific user class
        locust -f tests/test_load_testing.py ConcurrentAnalysisUser --host=http://localhost:8010
        
        # Run with specific parameters
        locust -f tests/test_load_testing.py -u 10 -r 2 -t 60s --host=http://localhost:8010
    """
    
    @staticmethod
    def light_load():
        """
        Light load scenario: 1-5 users, 1 user/second ramp-up
        Tests basic functionality under minimal load
        """
        return "-u 5 -r 1 -t 120s"
    
    @staticmethod 
    def moderate_load():
        """
        Moderate load scenario: 10-20 users, 2 users/second ramp-up
        Tests performance under typical usage
        """
        return "-u 20 -r 2 -t 300s"
    
    @staticmethod
    def heavy_load():
        """
        Heavy load scenario: 50+ users, 5 users/second ramp-up
        Tests system limits and concurrent processing
        """
        return "-u 50 -r 5 -t 600s"
    
    @staticmethod
    def spike_test():
        """
        Spike test: Rapid increase to high load
        Tests system resilience under sudden traffic spikes
        """
        return "-u 100 -r 10 -t 180s"


# Performance benchmarks for CI/CD validation
PERFORMANCE_REQUIREMENTS = {
    "health_check_max_ms": 100,
    "analysis_max_ms": 2000,  # 2 second max for single analysis
    "concurrent_analysis_max_ms": 1000,  # 1 second max under concurrent load
    "visualization_max_ms": 5000,  # 5 second max for visualization
    "max_error_rate_percent": 5,  # Maximum 5% error rate under load
    "min_throughput_rps": 10,  # Minimum 10 requests per second
}