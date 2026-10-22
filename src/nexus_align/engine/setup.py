"""Environment setup: distributed init, logging, seed, and config for train/eval."""

import os
import glob
from datetime import datetime

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import torch
import torch.distributed as dist

from nexus_align.engine.logger import init_log
from nexus_align.engine.tracker import init_wandb
from nexus_align.engine.distributed import init_dist_env
from nexus_align.utils.seed import set_seed, set_deterministic
from nexus_align.utils.provenance import collect_provenance


# Config sections that eval inherits from the training run.
INHERIT_SECTIONS = ("model", "algorithm")


def inherit_train_config(cfg) -> None:
    """Load the training run's saved config so eval uses the same model and algorithm."""
    config_path = os.path.join(cfg.log_dir, "config_train.yaml")
    if not os.path.exists(config_path):
        print(f"⚠️ WARNING: no training config at <{config_path}>; using the composed config")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = OmegaConf.create(yaml.safe_load(f))
    # Runs before the naming cleanup saved display names such as "meanflowsit-b/2".
    train_cfg.model.name = train_cfg.model.name.replace("-", "_").replace("/", "_")

    overrides = []
    for override in HydraConfig.get().overrides.task:
        override = override.lstrip("+")
        key = override.split("=")[0]
        if "." in key and key.split(".")[0] in INHERIT_SECTIONS:
            overrides.append(override)
    for section in INHERIT_SECTIONS:
        if section in train_cfg:
            cfg[section] = train_cfg[section]
    cfg.merge_with(OmegaConf.from_dotlist(overrides))
    print(f"✅ Inherited {INHERIT_SECTIONS} from <{config_path}>")


def resolve_eval_dir(cfg, timestamp: str) -> str:
    """Reuse the latest eval dir of this evaluator, or open a new one for this run."""
    eval_dir = [None]
    if dist.get_rank() == 0:
        eval_root = os.path.join(cfg.log_dir, "eval")
        prefix = cfg.evaluator.name
        if "data" in cfg.evaluator:  # one dir per prompt set, e.g. reward-drawbench-*
            prefix = f"{prefix}-{cfg.evaluator.data.name}"
        existing = sorted(glob.glob(os.path.join(eval_root, f"{prefix}-*")))
        if existing and not cfg.evaluator.new_run:
            eval_dir[0] = existing[-1]
        else:
            eval_dir[0] = os.path.join(eval_root, f"{prefix}-{timestamp}")
    dist.broadcast_object_list(eval_dir, src=0)
    return eval_dir[0]


def prepare_env(cfg) -> None:
    """Prepare environment: distributed init, logging, seed, config."""
    # Initialize distributed environment.
    world_size, rank = init_dist_env()
    base_seed = int(cfg.trainer.seed)

    # Update configs.
    is_train = cfg.job_type == "train"
    with open_dict(cfg):
        if not is_train and cfg.get("evaluator") is None:
            raise ValueError("❌ evaluator is required in eval mode, e.g. evaluator=in1k_50k_256_fid_is")
        if len(cfg.exp_info.strip()) == 0:
            if not is_train:
                raise ValueError("❌ exp_info is required in eval mode to locate the experiment")
            names = [cfg.data.name, cfg.model.name, cfg.algorithm.name]
            cfg.exp_info = "-".join(names)
        ts = [datetime.now().strftime("%Y%m%d_%H%M%S")]
        dist.broadcast_object_list(ts, src=0)
        if is_train and cfg.get("timestamp", True):
            cfg.exp_info = f"{cfg.exp_info}-{ts[0]}"
        cfg.project_path = os.getcwd()
        cfg.log_dir = os.path.abspath(os.path.join(cfg.log_dir, cfg.exp_info))
        if not is_train:
            inherit_train_config(cfg)
            cfg.evaluator.eval_dir = resolve_eval_dir(cfg, ts[0])
        cfg.world_size = world_size
        cfg.rank = rank
        cfg.data.seed = base_seed
    
    # Initialize logger and wandb.
    init_log(
        exp_info=cfg.exp_info,
        debug_console=cfg.trainer.debug,
        exp_dir=os.path.join(cfg.log_dir, f"logs-{cfg.job_type}-{ts[0]}"),
    )
    if is_train:
        init_wandb(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            dir=cfg.log_dir,
            job_type=cfg.job_type
        )

    # Fix seed.
    runtime_seed = base_seed
    if is_train:
        runtime_seed = base_seed + rank
    set_seed(runtime_seed)
    set_deterministic(cfg.trainer.deterministic)

    # Allow TF32 on matmul and cudnn for faster training on Ampere+ GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Record run provenance (code, software, hardware) and save configs.
    with open_dict(cfg):
        cfg.provenance = collect_provenance()
    if rank == 0:
        config_dir = cfg.log_dir if is_train else cfg.evaluator.eval_dir
        config_save_path = os.path.join(config_dir, f"config_{cfg.job_type}.yaml")
        os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(config_save_path):  # a reused eval dir keeps its original config
            container = OmegaConf.to_container(cfg, resolve=True)
            with open(config_save_path, "w", encoding="utf-8") as f:
                yaml.dump(container, f, sort_keys=True, allow_unicode=True, indent=2)
            print(f"💾 Saved configs to <{config_save_path}>")
