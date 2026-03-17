"""Abstract base for reward models."""

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
