"""Common environment setup for train and evaluation entry points."""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import pyinstrument
import torch
from omegaconf import OmegaConf, open_dict

from nexus_align.engine.distributed import init_dist_env, dist_safe_exit
from nexus_align.engine.logger import init_log, init_wandb
from nexus_align.utils.random import set_seed, set_deterministic


@dataclass
class EnvContext:
    """Context returned after environment preparation."""

    world_size: int
    rank: int
    device: torch.device
    cfg_dict: dict
    seed: int
    profiler: Optional[pyinstrument.Profiler]


def prepare_env(
    cfg,
    *,
    validator: Optional[Callable[[dict], list]] = None,
) -> EnvContext:
    """
    Prepare common environment for train/evaluation: dist init, logging, seed, config.

    Args:
        cfg: OmegaConf config object.
        validator: Optional callable(cfg_dict) -> list of missing keys. Exits if non-empty.

    Returns:
        EnvContext with world_size, rank, device, cfg_dict, seed, profiler.
    """
    # Initialize distributed environment
    world_size, rank, device = init_dist_env()
    
    # For time cost analysis
    profiler = None
    if rank == 0 and cfg.common.time_cost_analysis:
        profiler = pyinstrument.Profiler()
        profiler.start()
    
    # Update configs
    with open_dict(cfg):
        log_dir = cfg.log.log_dir
        cfg.common.project_path = os.getcwd().split(log_dir)[0]
        cfg.log.log_dir = os.path.join(cfg.common.project_path, log_dir)

    # Initialize logger and wandb
    init_log(
        exp_info=cfg.log.exp_info,
        debug_console=cfg.common.debug_console,
        exp_dir=os.path.join(cfg.log.log_dir, "logs"),
    )
    wandb_init = init_wandb(
        entity=cfg.log.wandb.entity,
        project=cfg.log.wandb.project,
        name=cfg.log.wandb.name,
        wandb_offline=cfg.log.wandb.wandb_offline or cfg.common.debug,
    )
    with open_dict(cfg):
        cfg.log.wandb.wandb_init = wandb_init

    # Fix seed
    seed = cfg.common.seed
    if cfg.name == "train":
        seed = cfg.common.seed + rank
        cfg.common.seed = seed
    set_seed(seed)
    set_deterministic(cfg.common.deterministic)

    # Save configs
    if rank == 0:
        config_save_path = os.path.join(cfg.log.log_dir, "config.yaml")
        OmegaConf.save(config=cfg, resolve=True, f=config_save_path)
        print(f"💾 Saved configs to <{config_save_path}>")

    cfg_dict = OmegaConf.to_container(cfg)
    if validator is not None:
        missing = validator(cfg_dict)
        if missing:
            dist_safe_exit(exit_code=1, message=f"❌ Missing config keys: {missing}")

    return EnvContext(
        world_size=world_size,
        rank=rank,
        device=device,
        cfg_dict=cfg_dict,
        seed=seed,
        profiler=profiler,
    )


def with_env_setup(
    *,
    validator: Optional[Callable[[dict], list]] = None,
):
    """
    Decorator that runs prepare_env before the main function and passes EnvContext.

    The decorated function must accept (cfg, env: EnvContext).
    """

    def decorator(main_fn):
        def wrapper(cfg):
            env = prepare_env(
                cfg,
                validator=validator,
            )
            return main_fn(cfg, env)

        return wrapper

    return decorator
