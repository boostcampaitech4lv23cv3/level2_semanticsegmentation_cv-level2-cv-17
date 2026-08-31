# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Union

import mmcv
import pytz
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.utils.config import ConfigDict
from rich.console import Console

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes

KST_TZ = pytz.timezone("Asia/Seoul")
GPU_ID = 0
SEED = 2022
PROJ_DIR = Path(__file__).parent.parent.parent
WORK_DIRS = PROJ_DIR / "work_dirs"
PTH_PREFIX = "iter_"
TIME_FORMAT = "%m/%d %H:%M"

console = Console(record=True)


class WandbResumeType(str, Enum):
    allow = "allow"
    must = "must"
    never = "never"
    auto = "auto"


def get_latest_checkpoint(work_dir: Path) -> Union[str, None]:
    latest_checkpoint_file = work_dir / "latest_checkpoint"
    if not latest_checkpoint_file.exists():
        return None

    with open(latest_checkpoint_file, encoding="utf8") as f:
        checkpoint_path = f.read()

    return checkpoint_path


def get_wandb_hook_index(cfg) -> Union[int, None]:
    try:
        wandb_hook_index = next(
            i
            for i, hook in enumerate(cfg.log_config.hooks)
            if hook.type == "MMSegWandbHook"
        )
    except Exception:
        return None

    return wandb_hook_index


def set_wandb_name(wandb_hook: ConfigDict, config_path: Path) -> None:
    name = config_path.stem.replace("_", " ")
    name = datetime.now(KST_TZ).strftime(f"{TIME_FORMAT} ") + name
    wandb_hook.init_kwargs.name = name


def set_wandb_resume_id(wandb_hook: ConfigDict, resume_wandb_id) -> None:
    wandb_hook.init_kwargs.resume = "allow"
    wandb_hook.init_kwargs.id = resume_wandb_id


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--load-from", help="the checkpoint file to load weights from")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--resume-wandb-id", help="the checkpoint file to resume from")
    parser.add_argument(
        "--resume-wandb-type", default="allow", help="wandb resuming behavior "
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff_seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        "not be supported in version v0.22.0. Override some settings in the "
        "used config, the key-value pair in xxx=yyy format will be merged "
        "into config file. If the value to be overwritten is a list, it "
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        "marks are necessary and that no white space is allowed.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="resume from the latest checkpoint automatically.",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options. "
            "--options will not be supported in version v0.22.0."
        )
    if args.options:
        warnings.warn(
            "--options is deprecated in favor of --cfg-options. "
            "--options will not be supported in version v0.22.0."
        )
        args.cfg_options = args.options

    if args.resume_from is not None and args.resume_wandb_id is None:
        raise ValueError("'--resume-from' should be with '--resume-wandb-id'")

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # ------

    config_filepath = Path(args.config)

    work_dir = WORK_DIRS / config_filepath.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    elif latest_checkpoint := get_latest_checkpoint(work_dir):
        cfg.resume_from = latest_checkpoint

        if latest_checkpoint and args.resume_wandb_id is None:
            raise ValueError(
                "'latest_checkpoint' is found. Please set '--resume-wandb-id'"
            )

    if wandb_hook_index := get_wandb_hook_index(cfg):
        wandb_hook = cfg.log_config.hooks[wandb_hook_index]
        set_wandb_name(wandb_hook, config_filepath)

        if args.resume_wandb_id:
            set_wandb_resume_id(wandb_hook, args.resume_wandb_id)

    # ----------

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            "`--gpus` is deprecated because we only support "
            "single GPU mode in non-distributed training. "
            "Use `gpus=1` now."
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed training. Use the first GPU "
            "in `gpu_ids` now."
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # # work_dir is determined in this priority: CLI > segment in file > filename
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     cfg.work_dir = args.work_dir
    # elif cfg.get("work_dir", None) is None:
    #     # use config filename as default work_dir if cfg.work_dir is None
    #     cfg.work_dir = osp.join(
    #         "./work_dirs", osp.splitext(osp.basename(args.config))[0]
    #     )
    if args.load_from is not None:
        cfg.load_from = args.load_from
    # if args.resume_from is not None:
    #     cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.work_dir = str(work_dir)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y-%m-%d %H;%M", time.localtime())
    log_file = work_dir / f"{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta = {
        "env_info": env_info,
        "seed": seed,
        "exp_name": config_filepath.stem,
    }

    model = build_segmentor(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            "SyncBN is only supported with DDP. To be compatible with DP, "
            "we convert SyncBN to BN. Please use dist_train.sh which can "
            "avoid this error."
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # val_dataset.pipeline = cfg.data.train.pipeline
        # AttributeError: 'ConfigDict' object has no attribute 'pipeline'
        # -> dataset/train.py에서 대신 설정하기
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f"{__version__}+{get_git_hash()[:7]}",
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
