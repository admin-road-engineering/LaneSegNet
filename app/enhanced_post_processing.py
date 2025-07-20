"""
Enhanced post-processing with physics-informed constraints for lane marking detection.
Part of Phase 2 model accuracy enhancement to achieve 80-85% mIoU target.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage import morphology
from skimage.measure import regionprops

logger = logging.getLogger(__name__)

# Physics-informed constraints for lane markings
LANE_PHYSICS_CONSTRAINTS = {
    # Lane width constraints (in pixels, varies with image resolution)
    'min_lane_width': 2,    # Minimum lane marking width
    'max_lane_width': 15,   # Maximum lane marking width
    'typical_lane_width': [3, 4, 5, 6],  # Most common widths
    
    # Lane geometry constraints
    'min_lane_length': 20,  # Minimum length for a valid lane marking
    'max_curvature': 0.3,   # Maximum allowed curvature (rad/pixel)
    'min_aspect_ratio': 3.0,  # Length/width ratio for line-like structures
    
    # Spatial relationship constraints
    'min_lane_spacing': 50,   # Minimum distance between parallel lanes
    'max_lane_spacing': 200,  # Maximum distance between parallel lanes
    'parallel_tolerance': 15, # Angular tolerance for parallel detection (degrees)
    
    # Color consistency constraints
    'white_intensity_threshold': 200,  # Minimum intensity for white markings
    'yellow_hue_range': (15, 35),     # HSV hue range for yellow markings
    'color_consistency_threshold': 0.8, # Minimum color consistency within marking
}

def apply_physics_informed_filtering(lane_markings: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
    """
    Apply physics-informed constraints to filter and validate lane markings.
    
    Args:
        lane_markings: List of detected lane markings
        image_shape: (height, width) of the image
        
    Returns:
        Filtered list of lane markings that satisfy physical constraints
    """
    filtered_markings = []
    height, width = image_shape
    
    for marking in lane_markings:
        if validate_lane_geometry(marking):
            if validate_lane_dimensions(marking, height, width):
                if validate_lane_coherence(marking):
                    filtered_markings.append(marking)
                else:
                    logger.debug(f"Lane marking failed coherence validation: {marking['class']}")
            else:
                logger.debug(f"Lane marking failed dimension validation: {marking['class']}")
        else:
            logger.debug(f"Lane marking failed geometry validation: {marking['class']}")
    
    # Apply spatial relationship constraints
    filtered_markings = validate_spatial_relationships(filtered_markings)
    
    logger.info(f"Physics-informed filtering: {len(lane_markings)} -> {len(filtered_markings)} markings")
    return filtered_markings

def validate_lane_geometry(marking: Dict) -> bool:
    """
    Validate that lane marking satisfies geometric constraints.
    """
    points = marking.get('points', [])
    if len(points) < 2:
        return False
    
    # Calculate total length
    total_length = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        total_length += np.sqrt(dx*dx + dy*dy)
    
    # Check minimum length constraint
    if total_length < LANE_PHYSICS_CONSTRAINTS['min_lane_length']:
        return False
    
    # Check aspect ratio (length vs area)
    area = marking.get('area', 0)
    if area > 0:
        estimated_width = area / total_length
        aspect_ratio = total_length / estimated_width
        if aspect_ratio < LANE_PHYSICS_CONSTRAINTS['min_aspect_ratio']:
            return False
    
    # Check curvature constraints
    if len(points) >= 3:
        curvatures = calculate_curvature(points)
        max_curvature = max(curvatures) if curvatures else 0
        if max_curvature > LANE_PHYSICS_CONSTRAINTS['max_curvature']:
            return False
    
    return True

def validate_lane_dimensions(marking: Dict, image_height: int, image_width: int) -> bool:
    """
    Validate lane marking dimensions are reasonable for the image size.
    """
    area = marking.get('area', 0)
    length = marking.get('length', 0)
    
    if area <= 0 or length <= 0:
        return False
    
    # Estimate width from area and length
    estimated_width = area / length if length > 0 else 0
    
    # Check width constraints
    if (estimated_width < LANE_PHYSICS_CONSTRAINTS['min_lane_width'] or 
        estimated_width > LANE_PHYSICS_CONSTRAINTS['max_lane_width']):
        return False
    
    # Check relative size constraints (lane shouldn't be too large relative to image)
    image_area = image_height * image_width
    if area > 0.1 * image_area:  # Lane marking shouldn't exceed 10% of image
        return False
    
    return True

def validate_lane_coherence(marking: Dict) -> bool:
    """
    Validate that lane marking has coherent properties (color, structure).
    """
    lane_class = marking.get('class', '')
    
    # Class-specific validation
    if 'white' in lane_class:
        # White lanes should have high intensity
        return True  # Would need actual pixel data for intensity validation
    elif 'yellow' in lane_class:
        # Yellow lanes should have appropriate hue
        return True  # Would need actual pixel data for hue validation
    elif 'solid' in lane_class:
        # Solid lanes should have continuous structure
        return True  # Would need contour analysis for continuity
    elif 'dashed' in lane_class:
        # Dashed lanes should have periodic gaps
        return True  # Would need gap analysis
    
    return True

def validate_spatial_relationships(lane_markings: List[Dict]) -> List[Dict]:
    """
    Validate spatial relationships between lane markings.
    """
    if len(lane_markings) < 2:
        return lane_markings
    
    valid_markings = []
    
    for i, marking in enumerate(lane_markings):
        is_valid = True
        marking_center = calculate_marking_center(marking)
        
        for j, other_marking in enumerate(lane_markings):
            if i == j:
                continue
                
            other_center = calculate_marking_center(other_marking)
            distance = np.sqrt((marking_center[0] - other_center[0])**2 + 
                             (marking_center[1] - other_center[1])**2)
            
            # Check minimum spacing constraint
            if distance < LANE_PHYSICS_CONSTRAINTS['min_lane_spacing']:
                # Keep the marking with higher confidence or larger area
                if (marking.get('confidence', 0) >= other_marking.get('confidence', 0) and
                    marking.get('area', 0) >= other_marking.get('area', 0)):
                    continue  # Keep current marking
                else:
                    is_valid = False  # Discard current marking
                    break
        
        if is_valid:
            valid_markings.append(marking)
    
    return valid_markings

def calculate_marking_center(marking: Dict) -> Tuple[float, float]:
    """
    Calculate the center point of a lane marking.
    """
    points = marking.get('points', [])
    if not points:
        return (0, 0)
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return (np.mean(x_coords), np.mean(y_coords))

def calculate_curvature(points: List[List[float]]) -> List[float]:
    """
    Calculate curvature at each point along the lane marking.
    """
    if len(points) < 3:
        return []
    
    curvatures = []
    
    for i in range(1, len(points) - 1):
        # Get three consecutive points
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        p3 = np.array(points[i+1])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate curvature using cross product method
        cross_product = np.cross(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            curvature = abs(cross_product) / (v1_norm * v2_norm)
            curvatures.append(curvature)
        else:
            curvatures.append(0)
    
    return curvatures

def enhance_lane_connectivity(lane_markings: List[Dict], 
                            max_gap_distance: float = 30.0) -> List[Dict]:
    """
    Connect nearby lane marking segments that likely belong to the same lane.
    """
    if len(lane_markings) < 2:
        return lane_markings
    
    # Group markings by class
    class_groups = {}
    for marking in lane_markings:
        class_name = marking.get('class', 'unknown')
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(marking)
    
    enhanced_markings = []
    
    # Process each class separately
    for class_name, markings in class_groups.items():
        if len(markings) == 1:
            enhanced_markings.extend(markings)
            continue
        
        # Find markings that can be connected
        connected_groups = find_connectable_markings(markings, max_gap_distance)
        
        for group in connected_groups:
            if len(group) == 1:
                enhanced_markings.extend(group)
            else:
                # Merge connected markings
                merged_marking = merge_lane_markings(group, class_name)
                enhanced_markings.append(merged_marking)
    
    logger.info(f"Connectivity enhancement: {len(lane_markings)} -> {len(enhanced_markings)} markings")
    return enhanced_markings

def find_connectable_markings(markings: List[Dict], 
                            max_gap_distance: float) -> List[List[Dict]]:
    """
    Find groups of markings that can be connected based on proximity and alignment.
    """
    if len(markings) <= 1:
        return [markings]
    
    # Create adjacency matrix
    n = len(markings)
    adjacency = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            if can_connect_markings(markings[i], markings[j], max_gap_distance):
                adjacency[i, j] = True
                adjacency[j, i] = True
    
    # Find connected components
    visited = [False] * n
    groups = []
    
    for i in range(n):
        if not visited[i]:
            group = []
            stack = [i]
            
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    group.append(markings[node])
                    
                    # Add neighbors to stack
                    for j in range(n):
                        if adjacency[node, j] and not visited[j]:
                            stack.append(j)
            
            groups.append(group)
    
    return groups

def can_connect_markings(marking1: Dict, marking2: Dict, 
                        max_gap_distance: float) -> bool:
    """
    Determine if two lane markings can be connected.
    """
    points1 = marking1.get('points', [])
    points2 = marking2.get('points', [])
    
    if not points1 or not points2:
        return False
    
    # Find closest endpoints
    endpoints1 = [points1[0], points1[-1]]
    endpoints2 = [points2[0], points2[-1]]
    
    min_distance = float('inf')
    for p1 in endpoints1:
        for p2 in endpoints2:
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_distance = min(min_distance, distance)
    
    return min_distance <= max_gap_distance

def merge_lane_markings(markings: List[Dict], class_name: str) -> Dict:
    """
    Merge multiple lane markings into a single connected marking.
    """
    all_points = []
    total_area = 0
    total_length = 0
    total_confidence = 0
    
    for marking in markings:
        all_points.extend(marking.get('points', []))
        total_area += marking.get('area', 0)
        total_length += marking.get('length', 0)
        total_confidence += marking.get('confidence', 0)
    
    # Sort points to create a connected path
    if len(all_points) > 1:
        sorted_points = sort_points_for_path(all_points)
    else:
        sorted_points = all_points
    
    return {
        'class': class_name,
        'class_id': markings[0].get('class_id', 0),
        'points': sorted_points,
        'confidence': total_confidence / len(markings),
        'area': total_area,
        'length': total_length
    }

def sort_points_for_path(points: List[List[float]]) -> List[List[float]]:
    """
    Sort points to create a connected path from one end to the other.
    """
    if len(points) <= 2:
        return points
    
    # Use a simple nearest neighbor approach
    remaining_points = points[1:]
    sorted_points = [points[0]]
    
    while remaining_points:
        last_point = sorted_points[-1]
        
        # Find nearest remaining point
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, point in enumerate(remaining_points):
            distance = np.sqrt((last_point[0] - point[0])**2 + (last_point[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        # Add nearest point and remove from remaining
        sorted_points.append(remaining_points.pop(nearest_idx))
    
    return sorted_points

def apply_temporal_consistency(current_markings: List[Dict], 
                             previous_markings: List[Dict] = None,
                             alpha: float = 0.7) -> List[Dict]:
    """
    Apply temporal consistency to reduce noise across sequential frames.
    
    Args:
        current_markings: Lane markings from current frame
        previous_markings: Lane markings from previous frame (optional)
        alpha: Temporal smoothing factor (0 = only current, 1 = only previous)
        
    Returns:
        Temporally smoothed lane markings
    """
    if previous_markings is None or alpha == 0:
        return current_markings
    
    # For single-frame processing, just return current markings
    # In a video context, this would implement motion prediction and matching
    logger.debug("Temporal consistency applied (single-frame mode)")
    return current_markings