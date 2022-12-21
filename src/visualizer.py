"""Visualize segmentation dataset with FifyOne.

References:
    - https://github.com/voxel51/fiftyone
    - https://voxel51.com/docs/fiftyone/
    - https://voxel51.com/docs/fiftyone/user_guide/config.html
"""
import fiftyone as fo
import typer
from fiftyone.types import COCODetectionDataset

fo.app_config.color_by = "label"

DATA_DIR = "/opt/ml/input/data"
ANNOTATION_FILENAME = "train_all.json"
PORT = 30001

IMAGE_IDS = None  # list of specific image IDs to load
LABEL_TYPES = ["detections", "segmentations"]
CLASSES = None  # list of specific classes to load


def main(
    data_path: str = typer.Option(DATA_DIR, help="image data path"),
    annotation_filename: str = typer.Option(
        ANNOTATION_FILENAME,
        help="annotation file path",
    ),
    include_id: bool = typer.Option(
        False, help="whether to include the COCO ID of each sample"
    ),
    include_annotation_id: bool = typer.Option(
        False,
        help="whether to include the COCO ID of each annotation",
    ),
    extra_attrs: bool = typer.Option(
        True,
        help="whether to load extra annotation attributes",
    ),
    max_samples: int = typer.Option(
        None,
        help="a maximum number of samples to load",
    ),
    shuffle: bool = typer.Option(
        False,
        help="whether to randomly shuffle",
    ),
    seed: int = typer.Option(None, help="a random seed to use when shuffling"),
):
    dataset = fo.Dataset.from_dir(
        dataset_type=COCODetectionDataset,
        data_path=data_path,
        labels_path=f"{data_path}/{annotation_filename}",
        label_types=LABEL_TYPES,
        # Optional arguments can be found on `COCODetectionDatasetImporter`
        # from fiftyone.utils.coco import COCODetectionDatasetImporter
        classes=CLASSES,
        image_ids=IMAGE_IDS,
        include_id=include_id,
        include_annotation_id=include_annotation_id,
        extra_attrs=extra_attrs,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
    )
    session = fo.launch_app(dataset, port=PORT, address="0.0.0.0")
    session.wait()


if __name__ == "__main__":
    typer.run(main)
