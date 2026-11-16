"""
NFT trainer: online RL on the forward process with a soft-updated sampling policy.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
            - scripts/train_nft_sd3.py: the epoch loop of main
"""

import numpy as np

from nexus_align.rewards.multi_reward import combine_group_zscores
from nexus_align.trainers.grpo_trainer import GRPOTrainer


class NFTTrainer(GRPOTrainer):
    """
    Generic trainer for negative-aware fine-tuning, with the epoch loop of GRPOTrainer.

    - The trainer trains on every sample, because a zero advantage still gives a loss
      (r = 0.5).
    - The trainer calls ema_step once after each optimizer step.
    - A multi-reward, that is a reward with weights, trains on the weighted mean of the
      group z-scores of its calibrated components, as DiffusionART and TwinTilt do,
      and not on the group z-scores of its summed score.
    - After each epoch, the algorithm soft-updates its sampling (old) policy before the
      trainer saves the checkpoint.

    The algorithm provides the GRPOTrainer interface and update_old_policy(epoch, step).
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
        self.algorithm.update_old_policy(self.epoch, self.total_step)
