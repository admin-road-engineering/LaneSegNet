#!/usr/bin/env python3
"""
Generate aerial viewpoint images using CARLA simulator.
Target: ~2k synthetic aerial road scenes for self-supervised pre-training.
"""

import os
import sys
import time
import json
import random
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CARLAAerialGenerator:
    def __init__(self, download_dir="data/unlabeled_aerial/carla_synthetic"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_count = 2000
        self.collected_count = 0
        
        # CARLA connection parameters
        self.carla_host = 'localhost'
        self.carla_port = 2000
        self.carla = None
        self.world = None
        self.camera = None
        
        # Aerial camera settings
        self.camera_configs = [
            {'height': 50, 'pitch': -90, 'fov': 90},   # Direct overhead
            {'height': 40, 'pitch': -85, 'fov': 85},   # Slight angle
            {'height': 60, 'pitch': -88, 'fov': 95},   # Higher altitude
            {'height': 35, 'pitch': -80, 'fov': 80},   # More angled
        ]
        
        # Weather variations for diversity
        self.weather_presets = [
            'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
            'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
            'ClearSunset', 'CloudySunset', 'WetSunset'
        ]
        
    def check_carla_connection(self):
        """Check if CARLA simulator is running and accessible."""
        try:
            import carla
            self.carla = carla
            
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(10.0)
            
            # Test connection
            world = client.get_world()
            logger.info(f"Connected to CARLA server: {world.get_map().name}")
            return True
            
        except ImportError:
            logger.error("CARLA Python API not installed. Please install CARLA 0.9.13+")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            logger.info("Make sure CARLA simulator is running on localhost:2000")
            return False
    
    def setup_world_and_camera(self):
        """Setup CARLA world and aerial camera."""
        try:
            client = self.carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(10.0)
            self.world = client.get_world()
            
            # Get camera blueprint
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Set camera attributes
            camera_bp.set_attribute('image_size_x', '1280')
            camera_bp.set_attribute('image_size_y', '1280')
            camera_bp.set_attribute('sensor_tick', '0.1')
            
            return camera_bp
            
        except Exception as e:
            logger.error(f"Failed to setup CARLA world: {e}")
            return None
    
    def get_random_spawn_point(self):
        """Get random spawn point from available spawn points."""
        spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(spawn_points)
    
    def set_weather(self, weather_preset):
        """Set weather conditions."""
        try:
            weather = getattr(self.carla.WeatherParameters, weather_preset)
            self.world.set_weather(weather)
            logger.debug(f"Set weather: {weather_preset}")
        except AttributeError:
            # Custom weather if preset not available
            weather = self.carla.WeatherParameters(
                cloudiness=random.uniform(0, 100),
                precipitation=random.uniform(0, 50),
                sun_altitude_angle=random.uniform(30, 90),
                sun_azimuth_angle=random.uniform(0, 360)
            )
            self.world.set_weather(weather)
            logger.debug("Set custom weather")
    
    def spawn_aerial_camera(self, spawn_point, config):
        """Spawn camera at aerial position."""
        try:
            camera_bp = self.setup_world_and_camera()
            if not camera_bp:
                return None
            
            # Calculate aerial position
            aerial_transform = self.carla.Transform(
                self.carla.Location(
                    x=spawn_point.location.x,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z + config['height']
                ),
                self.carla.Rotation(
                    pitch=config['pitch'],
                    yaw=spawn_point.rotation.yaw,
                    roll=0
                )
            )
            
            # Set FOV
            camera_bp.set_attribute('fov', str(config['fov']))
            
            # Spawn camera
            camera = self.world.spawn_actor(camera_bp, aerial_transform)
            time.sleep(0.5)  # Allow camera to initialize
            
            return camera
            
        except Exception as e:
            logger.error(f"Failed to spawn camera: {e}")
            return None
    
    def capture_image(self, camera, image_id):
        """Capture image from camera."""
        try:
            image_data = {'data': None, 'ready': False}
            
            def image_callback(image):
                """Callback function for image capture."""
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = np.reshape(array, (image.height, image.width, 4))
                # Convert BGRA to RGB
                array = array[:, :, [2, 1, 0]]
                image_data['data'] = array
                image_data['ready'] = True
            
            # Attach callback and capture
            camera.listen(image_callback)
            
            # Wait for image capture
            timeout = 5.0
            start_time = time.time()
            while not image_data['ready'] and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            camera.stop()
            
            if image_data['ready']:
                # Save image
                filename = f"carla_aerial_{image_id:06d}.jpg"
                filepath = self.download_dir / "processed" / filename
                filepath.parent.mkdir(exist_ok=True)
                
                img = Image.fromarray(image_data['data'])
                img.save(filepath, "JPEG", quality=90)
                
                return str(filepath)
            else:
                logger.warning(f"Timeout capturing image {image_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return None
    
    def generate_aerial_scenes(self):
        """Generate aerial scene images."""
        if not self.check_carla_connection():
            return []
        
        logger.info("Starting CARLA aerial scene generation...")
        
        collected_images = []
        
        with tqdm(total=self.target_count, desc="Generating aerial scenes") as pbar:
            while self.collected_count < self.target_count:
                try:
                    # Random spawn point
                    spawn_point = self.get_random_spawn_point()
                    
                    # Random camera configuration
                    config = random.choice(self.camera_configs)
                    
                    # Random weather
                    weather = random.choice(self.weather_presets)
                    self.set_weather(weather)
                    
                    # Spawn camera
                    camera = self.spawn_aerial_camera(spawn_point, config)
                    if not camera:
                        continue
                    
                    # Capture image
                    filepath = self.capture_image(camera, self.collected_count)
                    
                    # Cleanup camera
                    if camera:
                        camera.destroy()
                    
                    if filepath:
                        collected_images.append(filepath)
                        self.collected_count += 1
                        pbar.update(1)
                    
                    # Small delay between captures
                    time.sleep(0.5)
                    
                except KeyboardInterrupt:
                    logger.info("Generation interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error generating scene {self.collected_count}: {e}")
                    continue
        
        return collected_images
    
    def create_manifest(self, image_paths):
        """Create manifest file for generated images."""
        manifest = {
            'source': 'CARLA Simulator',
            'count': len(image_paths),
            'target_count': self.target_count,
            'camera_configs': self.camera_configs,
            'weather_presets': self.weather_presets,
            'images': image_paths,
            'processing_date': str(pd.Timestamp.now()),
            'description': 'Synthetic aerial road scenes from CARLA simulator for SSL pre-training'
        }
        
        manifest_path = self.download_dir / "carla_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created manifest: {manifest_path}")
    
    def generate_dataset(self):
        """Main method to generate aerial dataset."""
        try:
            collected_images = self.generate_aerial_scenes()
            
            if collected_images:
                self.create_manifest(collected_images)
                logger.info(f"CARLA generation complete: {len(collected_images)}/{self.target_count} images")
            else:
                logger.warning("No aerial scenes generated from CARLA")
            
            return collected_images
            
        except Exception as e:
            logger.error(f"Failed to generate CARLA dataset: {e}")
            return []

def main():
    """Main execution function."""
    generator = CARLAAerialGenerator()
    
    try:
        collected_images = generator.generate_dataset()
        
        if collected_images:
            print(f"✅ Successfully generated {len(collected_images)} aerial scenes from CARLA")
            print(f"📁 Images saved to: {generator.download_dir}/processed/")
            print(f"📄 Manifest: {generator.download_dir}/carla_manifest.json")
        else:
            print("❌ Failed to generate aerial scenes from CARLA")
            print("💡 Make sure CARLA simulator is running and accessible")
            
    except KeyboardInterrupt:
        print("\n⏹️ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    # Add pandas import for timestamp
    try:
        import pandas as pd
    except ImportError:
        import datetime as dt
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return dt.datetime.now().isoformat()
    
    main()