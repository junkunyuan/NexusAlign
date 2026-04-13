"""Config schema and validation helpers."""

from typing import Any

import torch

# Model/config related constants
DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# Top-level keys required for train/eval config validation
TRAIN_REQUIRED_TOP_LEVEL = (
    "common",
    "log",
    "data",
    "model",
    "algorithm",
    "reward_model",
    "ckpt_manager",
)
EVAL_REQUIRED_TOP_LEVEL = (
    "common", 
    "log", 
    "data", 
    "model", 
    "reward_model"
)

def get_required_keys(cfg: dict[str, Any], *keys: str) -> list[str]:
    """
    Return list of missing top-level keys in cfg.
    Useful for minimal validation before training.
    """
    missing = []
    for k in keys:
        if k not in cfg:
            missing.append(k)
    return missing


def validate_train_config(cfg: dict[str, Any]) -> list[str]:
    """Return list of missing top-level keys for training config."""
    return get_required_keys(cfg, *TRAIN_REQUIRED_TOP_LEVEL)


def validate_eval_config(cfg: dict[str, Any]) -> list[str]:
    """Return list of missing top-level keys for evaluation config.

    Accepts either ``reward_model`` (single) or ``reward_models`` (list).
    """
    missing = get_required_keys(cfg, *EVAL_REQUIRED_TOP_LEVEL)
    if "reward_model" in missing and "reward_models" in cfg:
        missing.remove("reward_model")
    return missing
