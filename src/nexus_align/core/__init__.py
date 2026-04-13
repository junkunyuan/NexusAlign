"""Core abstractions and registry."""

from nexus_align.core.registry import registry
from nexus_align.core.base_model import BaseModel
from nexus_align.core.base_pipeline import BaseTrainPipeline
from nexus_align.core.base_algorithm import BaseAlgorithm
from nexus_align.core.base_reward_model import BaseRewardModel
from nexus_align.core.base_dataset import (
    BaseTextDataset,
    BaseTextImageDataset,
)
from nexus_align.core.config import validate_train_config, validate_eval_config

__all__ = [
    "registry",
    "BaseModel",
    "BaseTrainPipeline",
    "BaseAlgorithm",
    "BaseRewardModel",
    "BaseTextDataset",
    "BaseTextImageDataset",
    "validate_train_config",
    "validate_eval_config",
]
