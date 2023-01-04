# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOTrashDataset(CustomDataset):
    """COCO-Trash dataset."""

    CLASSES = (
        "Background", "General trash", "Paper", "Paper pack", "Metal", "Glass",
        "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing",
    )
    PALETTE = (
        (128, 224, 128), (128, 62, 62), (30, 142, 30), (192, 0, 0),  (50, 50, 160),
        (0, 224, 224), (0, 0, 224), (192, 224, 0), (192, 224, 224), (192, 96, 0),
        (0, 224, 0),
    )

    def __init__(self, **kwargs) -> None:
        super(COCOTrashDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", **kwargs
        )
