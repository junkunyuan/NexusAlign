"""
AWM trainer: online RL that trains the flow matching loss weighted by advantages.

References:
    - AWM:
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py: the epoch loop of
              main, with no sample filtering, one optimizer step per epoch, and the
              updates of the two EMAs after it
"""

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class AWMTrainer(GRPOTrainer):
    """
    Generic trainer for AWM, with the epoch loop of GRPOTrainer.

    - The trainer trains on every sample, as the official code does. A sample with a
      zero advantage still adds to the KL losses.
    - The trainer calls ema_step only after an optimizer step.
    - After each epoch, the algorithm soft-updates the EMA reference of its KL, before
      the trainer saves the checkpoint.

    The algorithm provides the GRPOTrainer interface and update_reference(step).
    """

    ema_every_batch = False

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def end_epoch(self) -> None:
        self.algorithm.update_reference(self.total_step)
