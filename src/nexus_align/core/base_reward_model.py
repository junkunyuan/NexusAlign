"""Abstract base class for reward models."""

import os
from abc import ABC, abstractmethod

import torch

from nexus_align.core.config import DTYPE_MAP


class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.
    """

    def __init__(self, model_name: str, device: torch.device, kwargs: dict) -> None:
        self.model_name = model_name
        
        self.model_path = os.path.join(
            kwargs["common"]["data_and_model_dir"], 
            kwargs["reward_model"]["path"]
        )

        self.device = device
        self.amp_dtype = DTYPE_MAP[kwargs["reward_model"]["amp_dtype"]]
        self.model_dtype = DTYPE_MAP[kwargs["reward_model"]["model_dtype"]]
        self.mode = kwargs["reward_model"]["mode"]
        self.task = kwargs["reward_model"].get("task", "aesthetic")
        raw_dims = kwargs["reward_model"].get("eval_dimensions", None)
        if isinstance(raw_dims, str):
            self.eval_dimensions = [s.strip() for s in raw_dims.split(",")]
        elif isinstance(raw_dims, (list, tuple)):
            self.eval_dimensions = list(raw_dims)
        else:
            self.eval_dimensions = None

        raw_keys = kwargs["reward_model"].get("eval_keys", None)
        if isinstance(raw_keys, str):
            self.eval_keys = [s.strip() for s in raw_keys.split(",")]
        elif isinstance(raw_keys, (list, tuple)):
            self.eval_keys = list(raw_keys)
        else:
            self.eval_keys = None

        self.reward_key = kwargs["reward_model"].get("reward_key", None)

        fsdp_kwargs = kwargs["reward_model"].get("fsdp", {})
        self.fsdp_kwargs = {
            "strategy": fsdp_kwargs.get("fsdp_strategy", None),
            "cpu_offload": fsdp_kwargs.get("fsdp_cpu_offload", False),
        }

    def load_model(self) -> None:
        """Load model."""
        pass

    @abstractmethod
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """Compute reward scores for a batch of samples."""
        pass
