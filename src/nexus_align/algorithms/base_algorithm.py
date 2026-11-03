"""Base algorithm: runtime model binding, and the forward and sample interfaces."""

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base algorithm interface."""

    def __init__(self) -> None:
        self.model = None
        self.device = None

    def prepare(self, model, device) -> None:
        """Bind the runtime model and device that the trainer owns."""
        self.model = model
        self.device = device

    @abstractmethod
    def forward(self, data):
        """Run one algorithm-specific training step."""
        ...

    @abstractmethod
    def sample(self, data):
        """Run one algorithm-specific sampling step."""
        ...
