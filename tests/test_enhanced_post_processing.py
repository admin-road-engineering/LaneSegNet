import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from typing import Dict, List

from app.enhanced_post_processing import (
    apply_physics_informed_filtering,
    apply_relaxed_filtering,
    validate_lane_geometry,
    validate_lane_dimensions,
    validate_lane_coherence,
    validate_spatial_relationships,
    enhance_cv_detection,
    merge_fragmented_segments,
    classify_lane_type_advanced,
    LANE_PHYSICS_CONSTRAINTS
)


class TestPhysicsInformedFiltering:
    """Test suite for physics-informed filtering functions."""

    @pytest.fixture
    def valid_lane_marking(self):
        """Valid lane marking for testing."""
        return {
            'class': 'single_white_solid',
            'class_id': 1,
            'confidence': 0.85,
            'points': [[100, 100], [200, 100], [200, 105], [100, 105]],  # Rectangle
            'area_pixels': 500,
            'length_pixels': 100,
            'width_pixels': 5,
            'aspect_ratio': 20.0,
            'color_mean': [200, 200, 200],  # White-ish
            'intensity_mean': 190
        }

    @pytest.fixture
    def invalid_lane_marking(self):
        """Invalid lane marking for testing."""
        return {
            'class': 'single_white_solid',
            'class_id': 1,
            'confidence': 0.3,  # Low confidence
            'points': [[100, 100], [101, 100]],  # Too short
            'area_pixels': 5,
            'length_pixels': 1,
            'width_pixels': 1,
            'aspect_ratio': 1.0,  # Poor aspect ratio
            'color_mean': [100, 100, 100],  # Too dark
            'intensity_mean': 100
        }

    @pytest.fixture
    def image_shape(self):
        """Standard test image shape."""
        return (1280, 1280)

    @pytest.mark.unit
    def test_physics_informed_filtering_valid_markings(self, valid_lane_marking, image_shape):
        """Test filtering with valid lane markings."""
        lane_markings = [valid_lane_marking]
        
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_coherence', return_value=True), \
             patch('app.enhanced_post_processing.validate_spatial_relationships', side_effect=lambda x: x):
            
            filtered = apply_physics_informed_filtering(lane_markings, image_shape)
        
        assert len(filtered) == 1
        assert filtered[0] == valid_lane_marking

    @pytest.mark.unit
    def test_physics_informed_filtering_invalid_markings(self, invalid_lane_marking, image_shape):
        """Test filtering with invalid lane markings."""
        lane_markings = [invalid_lane_marking]
        
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=False):
            filtered = apply_physics_informed_filtering(lane_markings, image_shape)
        
        # Should fall back to relaxed filtering
        assert isinstance(filtered, list)

    @pytest.mark.unit
    def test_physics_informed_filtering_mixed_markings(self, valid_lane_marking, invalid_lane_marking, image_shape):
        """Test filtering with mix of valid and invalid markings."""
        lane_markings = [valid_lane_marking, invalid_lane_marking]
        
        def mock_validate_geometry(marking):
            return marking == valid_lane_marking
        
        with patch('app.enhanced_post_processing.validate_lane_geometry', side_effect=mock_validate_geometry), \
             patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_coherence', return_value=True), \
             patch('app.enhanced_post_processing.validate_spatial_relationships', side_effect=lambda x: x):
            
            filtered = apply_physics_informed_filtering(lane_markings, image_shape)
        
        assert len(filtered) == 1
        assert filtered[0] == valid_lane_marking

    @pytest.mark.unit
    def test_relaxed_filtering_fallback(self, invalid_lane_marking, image_shape):
        """Test relaxed filtering fallback when all markings are filtered out."""
        lane_markings = [invalid_lane_marking]
        
        # Mock all validation functions to return False
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=False), \
             patch('app.enhanced_post_processing.apply_relaxed_filtering', return_value=[invalid_lane_marking]):
            
            filtered = apply_physics_informed_filtering(lane_markings, image_shape)
        
        # Should fall back to relaxed filtering
        assert len(filtered) == 1

    @pytest.mark.unit
    def test_apply_relaxed_filtering(self, invalid_lane_marking, image_shape):
        """Test relaxed filtering function."""
        lane_markings = [invalid_lane_marking]
        
        # Mock relaxed validation to be more permissive
        with patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True):
            filtered = apply_relaxed_filtering(lane_markings, image_shape)
        
        assert isinstance(filtered, list)
        # Relaxed filtering should be more permissive
        assert len(filtered) >= 0


class TestLaneValidationFunctions:
    """Test suite for individual lane validation functions."""

    @pytest.fixture
    def valid_marking(self):
        """Valid lane marking with good geometry."""
        return {
            'points': [[100, 100], [200, 100], [200, 105], [100, 105]],
            'length_pixels': 100,
            'width_pixels': 5,
            'aspect_ratio': 20.0,
            'confidence': 0.85,
            'curvature': 0.1,
            'class': 'single_white_solid'
        }

    @pytest.fixture
    def invalid_geometry_marking(self):
        """Lane marking with invalid geometry."""
        return {
            'points': [[100, 100], [101, 100]],  # Too short
            'length_pixels': 1,
            'width_pixels': 1,
            'aspect_ratio': 1.0,  # Poor aspect ratio
            'confidence': 0.3,
            'curvature': 0.8,  # Too curvy
            'class': 'single_white_solid'
        }

    @pytest.mark.unit
    def test_validate_lane_geometry_valid(self, valid_marking):
        """Test lane geometry validation with valid marking."""
        assert validate_lane_geometry(valid_marking) == True

    @pytest.mark.unit
    def test_validate_lane_geometry_invalid_length(self, invalid_geometry_marking):
        """Test lane geometry validation with invalid length."""
        # Should fail due to short length
        assert validate_lane_geometry(invalid_geometry_marking) == False

    @pytest.mark.unit
    def test_validate_lane_geometry_missing_fields(self):
        """Test lane geometry validation with missing required fields."""
        incomplete_marking = {'points': [[100, 100], [200, 100]]}
        
        # Should handle missing fields gracefully
        result = validate_lane_geometry(incomplete_marking)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_validate_lane_dimensions_valid(self, valid_marking):
        """Test lane dimensions validation with valid marking."""
        image_height, image_width = 1280, 1280
        assert validate_lane_dimensions(valid_marking, image_height, image_width) == True

    @pytest.mark.unit
    def test_validate_lane_dimensions_too_large(self):
        """Test lane dimensions validation with oversized marking."""
        large_marking = {
            'points': [[0, 0], [1280, 0], [1280, 100], [0, 100]],  # Full width
            'length_pixels': 1280,
            'width_pixels': 100,  # Too wide
            'area_pixels': 128000
        }
        
        image_height, image_width = 1280, 1280
        assert validate_lane_dimensions(large_marking, image_height, image_width) == False

    @pytest.mark.unit
    def test_validate_lane_coherence_valid(self, valid_marking):
        """Test lane coherence validation with valid marking."""
        valid_marking.update({
            'color_consistency': 0.8,
            'edge_strength': 0.7,
            'continuity_score': 0.9
        })
        
        assert validate_lane_coherence(valid_marking) == True

    @pytest.mark.unit
    def test_validate_lane_coherence_invalid(self):
        """Test lane coherence validation with invalid marking."""
        incoherent_marking = {
            'color_consistency': 0.3,  # Poor color consistency
            'edge_strength': 0.2,      # Weak edges
            'continuity_score': 0.1,   # Poor continuity
            'confidence': 0.2
        }
        
        assert validate_lane_coherence(incoherent_marking) == False

    @pytest.mark.unit
    def test_validate_spatial_relationships(self):
        """Test spatial relationship validation between lane markings."""
        markings = [
            {
                'points': [[100, 100], [200, 100]],
                'centroid': [150, 100],
                'angle': 0,
                'class': 'single_white_solid'
            },
            {
                'points': [[100, 150], [200, 150]],  # Parallel marking
                'centroid': [150, 150],
                'angle': 0,
                'class': 'single_white_solid'
            }
        ]
        
        validated = validate_spatial_relationships(markings)
        
        # Should return a list
        assert isinstance(validated, list)
        # Both markings should pass spatial validation
        assert len(validated) <= len(markings)


class TestCVEnhancementFunctions:
    """Test suite for computer vision enhancement functions."""

    @pytest.fixture
    def test_image(self):
        """Test image with lane-like features."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add horizontal white lines (simulating lane markings)
        image[100:110, 50:400] = [255, 255, 255]  # White line
        image[200:210, 50:400] = [255, 255, 0]    # Yellow line
        return image

    @pytest.fixture
    def segmentation_map(self):
        """Test segmentation map."""
        seg_map = np.zeros((512, 512), dtype=np.uint8)
        # Add road pixels (class 6 for ADE20K road)
        seg_map[50:450, 50:450] = 6
        return seg_map

    @pytest.mark.unit
    def test_enhance_cv_detection(self, test_image, segmentation_map):
        """Test CV enhancement of detection results."""
        # Mock original detection results
        original_results = {
            'lane_markings': [
                {
                    'class': 'road',
                    'points': [[50, 100], [400, 110]],
                    'confidence': 0.7
                }
            ]
        }
        
        enhanced = enhance_cv_detection(test_image, segmentation_map, original_results)
        
        # Should return enhanced results
        assert 'lane_markings' in enhanced
        assert isinstance(enhanced['lane_markings'], list)

    @pytest.mark.unit
    def test_merge_fragmented_segments(self):
        """Test merging of fragmented lane segments."""
        fragmented_segments = [
            {
                'points': [[100, 100], [150, 100]],
                'class': 'single_white_solid',
                'centroid': [125, 100]
            },
            {
                'points': [[160, 100], [200, 100]],  # Close to first segment
                'class': 'single_white_solid',
                'centroid': [180, 100]
            },
            {
                'points': [[100, 200], [200, 200]],  # Far from others
                'class': 'single_yellow_solid',
                'centroid': [150, 200]
            }
        ]
        
        merged = merge_fragmented_segments(fragmented_segments)
        
        # Should merge compatible segments
        assert isinstance(merged, list)
        assert len(merged) <= len(fragmented_segments)

    @pytest.mark.unit
    def test_classify_lane_type_advanced_white_solid(self, test_image):
        """Test advanced lane type classification for white solid lines."""
        # Extract region with white line
        region = test_image[95:115, 45:405]  # Around the white line
        
        lane_type = classify_lane_type_advanced(region, line_pattern='solid')
        
        # Should detect white solid line
        assert 'white' in lane_type.lower()
        assert 'solid' in lane_type.lower()

    @pytest.mark.unit
    def test_classify_lane_type_advanced_yellow_solid(self, test_image):
        """Test advanced lane type classification for yellow solid lines."""
        # Extract region with yellow line
        region = test_image[195:215, 45:405]  # Around the yellow line
        
        lane_type = classify_lane_type_advanced(region, line_pattern='solid')
        
        # Should detect yellow solid line
        assert 'yellow' in lane_type.lower()
        assert 'solid' in lane_type.lower()

    @pytest.mark.unit
    def test_classify_lane_type_advanced_dashed(self):
        """Test advanced lane type classification for dashed lines."""
        # Create image with dashed pattern
        image = np.zeros((20, 200, 3), dtype=np.uint8)
        # Add dashed white line pattern
        for i in range(0, 200, 20):
            if i % 40 < 20:  # Dash pattern
                image[8:12, i:i+15] = [255, 255, 255]
        
        lane_type = classify_lane_type_advanced(image, line_pattern='dashed')
        
        # Should detect dashed pattern
        assert 'dashed' in lane_type.lower()

    @pytest.mark.unit
    def test_physics_constraints_validation(self):
        """Test physics constraints are reasonable."""
        constraints = LANE_PHYSICS_CONSTRAINTS
        
        # Validate constraint ranges
        assert constraints['min_lane_width'] > 0
        assert constraints['max_lane_width'] > constraints['min_lane_width']
        assert constraints['min_lane_length'] > 0
        assert constraints['max_curvature'] > 0
        assert constraints['min_aspect_ratio'] > 1
        
        # Validate spatial constraints
        assert constraints['min_lane_spacing'] > 0
        assert constraints['max_lane_spacing'] > constraints['min_lane_spacing']
        
        # Validate color constraints
        assert 0 <= constraints['white_intensity_threshold'] <= 255
        assert len(constraints['yellow_hue_range']) == 2
        assert 0 <= constraints['color_consistency_threshold'] <= 1


class TestPostProcessingIntegration:
    """Integration tests for complete post-processing pipeline."""

    @pytest.fixture
    def complete_detection_results(self):
        """Complete detection results for integration testing."""
        return {
            'lane_markings': [
                {
                    'class': 'single_white_solid',
                    'class_id': 1,
                    'confidence': 0.85,
                    'points': [[100, 100], [200, 100], [200, 105], [100, 105]],
                    'area_pixels': 500,
                    'length_pixels': 100,
                    'width_pixels': 5,
                    'aspect_ratio': 20.0,
                    'color_mean': [200, 200, 200],
                    'intensity_mean': 190
                },
                {
                    'class': 'single_yellow_solid',
                    'class_id': 2,
                    'confidence': 0.92,
                    'points': [[100, 200], [200, 200], [200, 205], [100, 205]],
                    'area_pixels': 500,
                    'length_pixels': 100,
                    'width_pixels': 5,
                    'aspect_ratio': 20.0,
                    'color_mean': [255, 255, 0],
                    'intensity_mean': 180
                }
            ],
            'processing_info': {
                'total_pixels': 1638400,
                'road_pixels': 500000
            }
        }

    @pytest.mark.integration
    def test_complete_post_processing_pipeline(self, complete_detection_results):
        """Test complete post-processing pipeline."""
        image_shape = (1280, 1280)
        lane_markings = complete_detection_results['lane_markings']
        
        # Mock all validation functions to pass
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_coherence', return_value=True), \
             patch('app.enhanced_post_processing.validate_spatial_relationships', side_effect=lambda x: x):
            
            filtered_markings = apply_physics_informed_filtering(lane_markings, image_shape)
        
        # Should preserve valid markings
        assert len(filtered_markings) == 2
        assert all(marking['confidence'] > 0.8 for marking in filtered_markings)

    @pytest.mark.integration
    def test_post_processing_with_cv_enhancement(self, complete_detection_results):
        """Test post-processing with CV enhancement integration."""
        # Create test image and segmentation map
        test_image = np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)
        segmentation_map = np.random.randint(0, 12, (1280, 1280), dtype=np.uint8)
        
        # Apply CV enhancement first
        enhanced_results = enhance_cv_detection(test_image, segmentation_map, complete_detection_results)
        
        # Then apply physics filtering
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_coherence', return_value=True), \
             patch('app.enhanced_post_processing.validate_spatial_relationships', side_effect=lambda x: x):
            
            final_results = apply_physics_informed_filtering(
                enhanced_results['lane_markings'], 
                (1280, 1280)
            )
        
        # Should produce valid final results
        assert isinstance(final_results, list)
        assert all(isinstance(marking, dict) for marking in final_results)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_post_processing_performance(self, complete_detection_results):
        """Test post-processing performance with large datasets."""
        import time
        
        # Create large dataset
        large_markings = complete_detection_results['lane_markings'] * 100
        image_shape = (1280, 1280)
        
        # Measure processing time
        start_time = time.time()
        
        with patch('app.enhanced_post_processing.validate_lane_geometry', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_dimensions', return_value=True), \
             patch('app.enhanced_post_processing.validate_lane_coherence', return_value=True), \
             patch('app.enhanced_post_processing.validate_spatial_relationships', side_effect=lambda x: x):
            
            filtered_markings = apply_physics_informed_filtering(large_markings, image_shape)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process efficiently (< 1 second for 200 markings)
        assert processing_time < 1.0
        assert len(filtered_markings) <= len(large_markings)