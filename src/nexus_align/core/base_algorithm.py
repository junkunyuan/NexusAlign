"""Abstract base for algorithms."""

from typing import Any
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    Abstract base class for algorithms.
    """

    # -------------------------------------------------------------------------
    # Methods: data -> (rollout) -> (reward) -> (advantage) -> train
    # -------------------------------------------------------------------------
    @abstractmethod
    def prepare_data(self, data: dict) -> dict:
        """
        Prepare data from dataloader batch.

        All algorithms must override this method.
        GRPO needs to add "keys_to_build_groups" to the returned data dict to 
        indicate which keys should be duplicated for building GRPO groups.

        Args:
            data (`dict`): batch from dataloader (e.g. "text", "image").

        Returns:
            data (`dict`): prepared data.
        """
        ...

    def sample_responses(self, data: dict) -> dict:
        """
        Rollout: generate responses from the prepared data.

        Algorithm must override this method: GRPO.
        It must return with at least the key "reward_inputs" for compute_rewards.

        Args:
            data (`dict`): prepared data.

        Returns:
            data (`dict`): data with responses added.
        """
        ...

    def compute_rewards(self, data: dict) -> dict:
        """
        Score responses with reward model.
        
        Algorithm must override this method: GRPO.
        It must return with at least the key "rewards" for compute_advantages.

        Args:
            data (`dict`): data with responses added by sample_responses.

        Returns:
            data (`dict`): data with rewards added.
        """
        ...

    def compute_advantages(self, data: dict) -> dict:
        """
        Turn rewards into advantages.
        
        Algorithm must override this method: GRPO.
        It must return with at least the key "advantages" for train_one_step.

        Args:
            data (`dict`): data with rewards added by compute_rewards.

        Returns:
            data (`dict`): data with advantages added.
        """
        ...

    @abstractmethod
    def train_one_step(self, data: dict, **kwargs: Any) -> None:
        """
        Compute loss and update the model for one step.

        All algorithms must override this method.

        Args:
            data (`dict`): data to train on.
        
        Returns:
            None
        """
        ...
