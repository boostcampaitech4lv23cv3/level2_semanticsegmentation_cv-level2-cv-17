import os
import warnings

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

dataset_dir = "/opt/ml/input/data"
train_path = "/opt/ml/input/data/train.json"
val_path = "/opt/ml/input/data/val.json"
test_path = "/opt/ml/input/data/test.json"
category_names = [
    "Backgroud",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_dir, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx["area"], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def collate_fn(batch):
    return tuple(zip(*batch))


def define_transform():
    """Augmentation 적용 시 수정"""
    train_transform = A.Compose(
        [
        #A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=5, holes_number_y=5),       
        ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
        #A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=5, holes_number_y=5),       
        ToTensorV2()
        ]
    )

    test_transform = A.Compose(             # tta
        [   
            A.HorizontalFlip(),
            A.RandomBrightness(limit=0.5),
            A.RandomRotate90(),
            ToTensorV2()
        ]
    )

    return train_transform, val_transform, test_transform


def make_dataloader(batch_size=4):
    train_transform, val_transform, test_transform = define_transform()

    # train dataset
    train_dataset = CustomDataLoader(
        data_dir=train_path, mode="train", transform=train_transform
    )

    # validation dataset
    val_dataset = CustomDataLoader(
        data_dir=val_path, mode="val", transform=val_transform
    )

    # test dataset
    test_dataset = CustomDataLoader(
        data_dir=test_path, mode="test", transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
