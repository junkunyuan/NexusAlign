"""
NFT trainer: online RL on the forward process with a soft-updated sampling policy.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
            - scripts/train_nft_sd3.py: the epoch loop of main
"""

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class NFTTrainer(GRPOTrainer):
    """
    Generic trainer for negative-aware fine-tuning, with the epoch loop of GRPOTrainer.

    - The trainer trains on every sample, because a zero advantage still gives a loss
      (r = 0.5).
    - The trainer calls ema_step once after each optimizer step.
    - After each epoch, the algorithm soft-updates its sampling (old) policy before the
      trainer saves the checkpoint.

    The algorithm provides the GRPOTrainer interface and update_old_policy(epoch, step).
    """

    ema_every_batch = False

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def end_epoch(self) -> None:
        self.algorithm.update_old_policy(self.epoch, self.total_step)
