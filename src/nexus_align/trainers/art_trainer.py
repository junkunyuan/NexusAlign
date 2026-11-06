"""
ART trainer: online RL that regresses rollout velocities to a shared reward guidance.

References:
    - DiffusionART:
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/backends/sd3.py: run and _art_update, for the guidance of
              complete groups and the two disjoint updates per rollout
            - src/visual_rl/training/objectives.py: prompt_disjoint_windows
"""

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class ARTTrainer(GRPOTrainer):
    """
    Generic trainer for DiffusionART, with the epoch loop of GRPOTrainer.

    - Each optimizer step of an epoch gets an equal part of every prompt group.
    - The algorithm calculates the guidance of each group from the rollouts of all
      ranks, before the trainer splits the samples into optimizer steps.
    - The trainer trains on every sample. A sample of a group with equal rewards has
      no loss, but it stays in the mean, as in the official code.
    - The trainer calls ema_step only after an optimizer step.

    The algorithm provides the GRPOTrainer interface, and compute_guidance(samples)
    that returns the guidance of each sample.
    """

    ema_every_batch = False
    split_groups = True

    def sample_epoch(self) -> tuple[dict, dict]:
        samples, metrics = super().sample_epoch()
        samples["guidance"] = self.algorithm.compute_guidance(samples)
        return samples, metrics

    def filter_samples(self, samples: dict) -> dict:
        return samples
