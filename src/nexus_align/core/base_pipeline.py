"""
Abstract base for training pipelines.
Pipelines prepare data and sample responses for alignment training.
Used by algorithms that implement the prepare_data -> sample_responses interface.
"""

from abc import ABC, abstractmethod

from typing import Any


class BaseTrainPipeline(ABC):
    """
    Abstract base class for rollout pipelines.
    Implementations provide prepare_data and sample_responses.
    The responses dict must contain at least "reward_inputs" for the reward model.
    """

    @abstractmethod
    def prepare_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Build per-step inputs (e.g. prompts, embeddings) from dataloader batch.

        Args:
            data: batch from dataloader (e.g. "text", "image") and optional train_state.

        Returns:
            conditions dict passed to sample_responses.
        """
        pass

    @abstractmethod
    def sample_responses(self, conditions: dict[str, Any]) -> dict[str, Any]:
        """
        Rollout: generate responses (e.g. images) from conditions.

        Args:
            conditions: output of prepare_data.

        Returns:
            responses dict with at least the key "reward_inputs" for compute_rewards.
        """
        pass
