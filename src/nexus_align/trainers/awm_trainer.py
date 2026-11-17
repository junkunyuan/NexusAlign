"""
AWM trainer: online RL that trains the flow matching loss weighted by advantages.

References:
    - AWM:
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py: the epoch loop of
              main, with no sample filtering, one optimizer step per epoch, and the
              updates of the two EMAs after it
"""

import numpy as np

from nexus_align.rewards.multi_reward import combine_group_zscores
from nexus_align.trainers.grpo_trainer import GRPOTrainer


class AWMTrainer(GRPOTrainer):
    """
    Generic trainer for AWM, with the epoch loop of GRPOTrainer.

    - The trainer trains on every sample, as the official code does. A sample with a
      zero advantage still adds to the KL losses.
    - The trainer calls ema_step only after an optimizer step.
    - A multi-reward, that is a reward with weights, trains on the weighted mean of the
      group z-scores of its calibrated components, as DiffusionART does, so that the
      multi-reward runs compare. The official code normalizes the summed score.
    - After each epoch, the algorithm soft-updates the EMA reference of its KL, before
      the trainer saves the checkpoint.

    The algorithm provides the GRPOTrainer interface and update_reference(step).
    """

    ema_every_batch = False

    def compute_advantages(self, prompts: list[str], rewards: dict) -> np.ndarray:
        weights = getattr(self.reward, "weights", None)
        if weights is None:
            return super().compute_advantages(prompts, rewards)
        # The combined z-scores are normalized in each group already, so they are the
        # advantages, without the group centering and the global std of a single reward.
        scores = {key: rewards[f"{key}_calibrated"] for key in weights}
        return combine_group_zscores(prompts, scores, weights)

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def end_epoch(self) -> None:
        self.algorithm.update_reference(self.total_step)
