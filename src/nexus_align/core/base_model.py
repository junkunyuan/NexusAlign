"""Abstract base for models."""

from abc import ABC, abstractmethod

from torch import nn
from torch.nn import Parameter


class BaseModel(ABC):
    """
    Abstract base class for models.
    """

    @abstractmethod
    def get_trainable_module(self) -> nn.Module:
        """Return the main trainable module."""
        pass

    @abstractmethod
    def get_trainable_params(self) -> list[Parameter]:
        """Return trainable parameters."""
        pass
