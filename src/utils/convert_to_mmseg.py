"""COCO Segmentation 포맷의 데이터를 MMSegmentation에 맞게 변경하기

최종 구조는 아래와 같이 저장될 예정
    ├── /opt/ml/input/mmseg  (= mmseg_data_dir)
    │   ├── trash  (= dataset_name)
    │   │   ├── img_dir
    │   │   │   ├── train
    │   │   │   │   ├── 0001.jpg
    │   │   │   │   ├── 0002.jpg
    │   │   │   ├── val
    │   │   │   ├── test
    │   │   ├── ann_dir
    │   │   │   ├── train
    │   │   │   │   ├── 0001.png
    │   │   │   │   ├── 0001_color.png
    │   │   │   │   ├── 0002.png
    │   │   │   │   ├── 0002_color.png
    │   │   │   ├── val

Examples:
    $ python ./src/utils/convert_to_mmseg.py
    $ python ./src/utils/convert_to_mmseg.py --coco-file val.json
    $ python ./src/utils/convert_to_mmseg.py --coco-file=test.json
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from pycocotools.coco import COCO
from rich.console import Console
from rich.progress import track

INPUT_DIR = Path("/opt/ml/input")
DATA_ROOT = INPUT_DIR / "data"
MMSEG_DATA_ROOT = INPUT_DIR / "mmseg"

CLASSES = [
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
COLORS = [
    (0, 0, 0),
    (128, 224, 128),
    (128, 62, 62),
    (30, 142, 30),
    (192, 0, 0),
    (50, 50, 160),
    (0, 224, 224),
    (0, 0, 224),
    (192, 224, 0),
    (192, 224, 224),
    (192, 96, 0),
    (0, 224, 0),
]

console = Console()


def save_image(image: np.ndarray, path: Path):
    cv2.imwrite(str(path), image)


def copy_images(coco_json_path: Path, output_directory: Path) -> None:
    """Copy images from COCO Segmentation file

    Args:
        coco_json_path (Path): COCO Segmentation filepath
        directory (Path): output directory
    """
    console.rule("copy images")

    coco = COCO(coco_json_path)
    coco_directory = coco_json_path.parent

    for img in track(coco.imgs.values()):
        img_path = coco_directory / img["file_name"]
        output_path = output_directory / f"{img['id']:04}{img_path.suffix}"
        shutil.copyfile(img_path, output_path)


def create_masks(coco_json_path: Path, output_directory: Path) -> None:
    """Create masks from COCO Segmentation file

    Args:
        coco_json_path (Path): COCO Segmentation filepath
        directory (Path): output directory
    """
    console.rule("create masks")

    coco = COCO(coco_json_path)

    for img_id in track(coco.getImgIds()):
        img_info = coco.imgs[img_id]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            console.log(f"[WARNING] empty annotation with image_id: {img_id}")
            # continue

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        color_mask = np.zeros(
            (img_info["height"], img_info["width"], 3), dtype=np.uint8
        )

        for ann in anns:
            class_index = ann["category_id"]
            mask[coco.annToMask(ann) == 1] = class_index
            color_mask[coco.annToMask(ann) == 1] = COLORS[class_index]

        save_image(mask, output_directory / f"{img_id:04}.png")
        save_image(color_mask, output_directory / f"{img_id:04}_color.png")


def main(
    coco_file: str = typer.Option("train.json", help="COCO annotation 파일명"),
    dataset_name: str = typer.Option("trash", help="데이터셋 이름"),
    data_root: Path = typer.Option(DATA_ROOT, help="COCO annotation 파일이 위치한 디렉토리"),
    mmseg_data_dir: Path = typer.Option(
        MMSEG_DATA_ROOT, help="MMSegmentation에 맞게 변경하여 저장할 디렉토리"
    ),
) -> None:
    console.log("Convert from COCO to MMSegmentation", log_locals=True)

    coco_json_path = data_root / coco_file

    step = coco_json_path.stem
    assert step in ("train", "val", "test")
    console.log(f"step: {step!r}")

    img_dir = mmseg_data_dir / dataset_name / "img_dir" / step
    img_dir.mkdir(parents=True, exist_ok=True)
    copy_images(coco_json_path, img_dir)

    if step in ("train", "val"):
        ann_dir = mmseg_data_dir / dataset_name / "ann_dir" / step
        ann_dir.mkdir(parents=True, exist_ok=True)
        create_masks(coco_json_path, ann_dir)

    console.log("Done")


if __name__ == "__main__":
    typer.run(main)
