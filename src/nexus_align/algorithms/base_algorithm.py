"""Base algorithm: shared training config, forward and sample interfaces."""

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base algorithm interface."""

    def __init__(self, cfg_algorithm) -> None:
        self.cfg_algorithm = cfg_algorithm
        self.model = None
        self.device = None

    def prepare(self, model, device) -> None:
        """Bind runtime model and device owned by the trainer."""
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
