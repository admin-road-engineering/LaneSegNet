from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class AELDataset(BaseSegDataset):
    """Aerial Lane (AEL) Dataset for comprehensive lane marking detection."""
    
    METAINFO = {
        'classes': (
            'background',
            'white_solid',
            'white_dashed', 
            'yellow_solid'
        ),
        'palette': [
            [0, 0, 0],        # background (black)
            [255, 255, 255],  # white_solid (white)
            [200, 200, 200],  # white_dashed (light gray)
            [255, 255, 0]     # yellow_solid (yellow)
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