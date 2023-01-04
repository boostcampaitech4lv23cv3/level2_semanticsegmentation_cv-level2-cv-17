import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

HOME_DIR = Path("/opt/ml/")
INPUT_DIR = HOME_DIR / "input"
DATA_DIR = INPUT_DIR / "data"
PROJECT_DIR = HOME_DIR / "level2_semanticsegmentation_cv-level2-cv-17"
SAMPLE_PATH = PROJECT_DIR / "src" / "submission" / "sample_submission.csv"
MMSEG_DIR = PROJECT_DIR / "mmsegmentation"
WORK_DIRS = PROJECT_DIR / "work_dirs"
TEST_JSON_PATH = DATA_DIR / "test.json"


def get_latest_pth(work_dir: Path) -> Path:
    # latest.pth -> iter_20000.pth
    latest_file = work_dir / "latest.pth"
    if not latest_file.exists():
        raise ValueError("last_file is not found.")

    return latest_file.resolve() if latest_file.is_symlink() else latest_file


def get_latest(work_dir: Path) -> Union[str, None]:
    latest_file = work_dir / "latest"
    if not latest_file.exists():
        return None

    with open(latest_file, "r", encoding="utf8") as f:
        path = f.read()

    return path


def get_last_checkpoint(work_dir: Path) -> Union[str, None]:
    latest_checkpoint_file = work_dir / "last_checkpoint"
    if not latest_checkpoint_file.exists():
        return None

    with open(latest_checkpoint_file, "r", encoding="utf8") as f:
        checkpoint_path = f.read()

    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("config_name", help="test config name")

    args = parser.parse_args()

    work_dir = WORK_DIRS / args.config_name
    if not work_dir.exists() or not work_dir.is_dir():
        raise ValueError("work_dir is not found.")

    last_file = get_latest_pth(work_dir)

    config_file = work_dir / f"{args.config_name}.py"

    output_dir = work_dir / last_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    args.work_dir = work_dir
    args.output_dir = output_dir
    args.last_file = last_file
    args.config_file = config_file

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config_file)

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    # checkpoint = load_checkpoint(model, str(args.last_file), map_location="cpu")
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader)

    # submission 양식에 맞게 output 후처리
    input_size = 512
    output_size = 256

    submission = pd.read_csv(SAMPLE_PATH, index_col=None)

    with open(TEST_JSON_PATH, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # PredictionString 대입
    for image_id, predict in enumerate(output):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]

        predict = predict.reshape(1, 512, 512)
        mask = (
            predict.reshape(
                (
                    1,
                    output_size,
                    input_size // output_size,
                    output_size,
                    input_size // output_size,
                )
            )
            .max(4)
            .max(2)
        )  # resize to 256*256
        temp_mask = [mask]
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size**2]).astype(int)
        string = oms.flatten()
        submission = pd.concat(
            [
                submission,
                pd.DataFrame(
                    [
                        {
                            "image_id": file_name,
                            "PredictionString": " ".join(
                                str(e) for e in string.tolist()
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    submission.to_csv(args.output_dir / "submission.csv", index=False)
    submission.to_csv(
        args.output_dir / f"{args.config_name}-submission.csv", index=False
    )


if __name__ == "__main__":
    main()
