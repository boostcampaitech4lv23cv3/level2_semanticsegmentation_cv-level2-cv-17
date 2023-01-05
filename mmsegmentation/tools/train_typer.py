import copy
import os
import time
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Union

import pytz
import torch
import torch.distributed as dist
import typer
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, get_git_hash
from mmcv.utils.config import ConfigDict
from rich.console import Console
from typer import Argument, Option

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


class LauncherEnum(str, Enum):
    none = "none"
    pytorch = "pytorch"
    slurm = "slurm"
    mpi = "mpi"


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


def main(
    config: Path = Argument(
        ...,
        help="train config file path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    # work_dir: Path = Option(None, help="the dir to save logs and models"),
    load_from: str = Option(None, help="the checkpoint file to load weights from"),
    # resume_from: str = Option(None, help="the checkpoint file to resume from"),
    resume_wandb_id: str = Option(None, help="wandb run id to resume on wandb"),
    resume_wandb_type: WandbResumeType = Option(
        WandbResumeType.allow, help="wandb resuming behavior"
    ),
    no_validate: bool = Option(True, help="whether not to evaluate during training"),
    gpu_id: int = Option(GPU_ID, help="id of gpu to use"),
    seed: int = Option(SEED, help="random seed"),
    diff_seed: bool = Option(True, help="set different seeds for different ranks"),
    deterministic: bool = Option(True, help="set deterministic for CUDNN backend"),
    launcher: LauncherEnum = Option(LauncherEnum.none, help="job launcher"),
    local_rank: int = 0,
    # auto_resume: bool = Option(
    #     True, help="resume from the latest checkpoint automatically"
    # ),
    cfg_options: str = Option(
        None,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    ),
):
    console.log("train.py", log_locals=True)

    # enum to value
    launcher = launcher.value
    resume_wandb_type = resume_wandb_type.value

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank)

    cfg = Config.fromfile(config)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # ------

    config_filepath = Path(config)

    work_dir = WORK_DIRS / config_filepath.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    if latest_checkpoint := get_latest_checkpoint(work_dir):
        cfg.resume_from = latest_checkpoint

        if latest_checkpoint and resume_wandb_id is None:
            raise typer.BadParameter(
                "'latest_checkpoint' is found. Please set 'resume_wandb_id'"
            )

    if wandb_hook_index := get_wandb_hook_index(cfg):
        wandb_hook = cfg.log_config.hooks[wandb_hook_index]
        set_wandb_name(wandb_hook, config_filepath)

        if resume_wandb_id:
            set_wandb_resume_id(wandb_hook, resume_wandb_id)

    # ----------

    if load_from is not None:
        cfg.load_from = load_from

    cfg.gpu_ids = [gpu_id]

    cfg.auto_resume = True  # auto_resume

    # init distributed env first, since logger depends on the dist info.
    if launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # dump config
    cfg.dump(str(work_dir / config_filepath.name))

    # init the logger before other steps
    timestamp = time.strftime("%Y-%m-%d %H;%M", time.localtime())
    log_file = work_dir / f"{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # FIXME: logger에 RichHandler 추가하면?

    # set multi-process settings
    setup_multi_processes(cfg)

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
    seed = init_random_seed(seed, device=cfg.device)
    seed = seed + dist.get_rank() if diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {deterministic}")
    set_random_seed(seed, deterministic=deterministic)
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
        val_dataset.pipeline = cfg.data.train.pipeline
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
        validate=(not no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    typer.run(main)
