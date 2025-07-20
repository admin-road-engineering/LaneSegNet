from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class AELDataset(BaseSegDataset):
    """Aerial Lane (AEL) Dataset."""
    
    METAINFO = {
        'classes': ('background', 'single_white_solid', 'single_white_dashed'),
        'palette': [[0, 0, 0], [255, 255, 255], [128, 128, 128]] 
        # Background (black), SWS (white), SWD (gray)
        # You can adjust the palette colors as you like.
    }

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs) 