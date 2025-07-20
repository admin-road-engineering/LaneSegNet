from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class AELDataset(BaseSegDataset):
    """Aerial Lane (AEL) Dataset for comprehensive lane marking detection."""
    
    METAINFO = {
        'classes': (
            'background',
            'single_white_solid',
            'single_white_dashed', 
            'single_yellow_solid',
            'single_yellow_dashed',
            'double_white_solid',
            'double_yellow_solid',
            'road_edge',
            'center_line',
            'lane_divider',
            'crosswalk',
            'stop_line'
        ),
        'palette': [
            [0, 0, 0],        # background (black)
            [255, 255, 255],  # single_white_solid (white)
            [200, 200, 200],  # single_white_dashed (light gray)
            [255, 255, 0],    # single_yellow_solid (yellow)
            [200, 200, 0],    # single_yellow_dashed (dark yellow)
            [255, 255, 255],  # double_white_solid (white)
            [255, 255, 0],    # double_yellow_solid (yellow)
            [0, 255, 0],      # road_edge (green)
            [255, 0, 0],      # center_line (red)
            [255, 0, 255],    # lane_divider (magenta)
            [0, 255, 255],    # crosswalk (cyan)
            [255, 165, 0]     # stop_line (orange)
        ]
    }

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs) 