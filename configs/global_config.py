# Central configuration for global constants to ensure consistency across the codebase.

# Number of classes: Matches current dataset ([0: background, 1: lane_type1, 2: lane_type2]).
# TODO: Update to 12 for class expansion phase (see app/model_loader.py LANE_CLASSES)
NUM_CLASSES = 3

# Image processing constants
IMG_SIZE = 512

# Training constants
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-4

# Class names for 3-class system
CLASS_NAMES = [
    'background',
    'lane_type1', 
    'lane_type2'
]

# Future expansion: When scaling to 12 classes, update NUM_CLASSES and CLASS_NAMES
# LANE_CLASSES_12 = [
#     'background', 'single_white_solid', 'single_white_dashed', 'single_yellow_solid',
#     'single_yellow_dashed', 'double_white_solid', 'double_yellow_solid', 'road_edge',
#     'center_line', 'lane_divider', 'crosswalk', 'stop_line'
# ]