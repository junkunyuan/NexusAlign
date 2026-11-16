"""
ART trainer: online RL that regresses rollout velocities to a shared reward guidance.

References:
    - DiffusionART:
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/backends/sd3.py: run and _art_update, for the guidance of
              complete groups and the two disjoint updates per rollout;
              _art_guidance_scores, for the training reward of a multi-reward
            - src/visual_rl/training/objectives.py: prompt_disjoint_windows
"""

import numpy as np
import torch

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class ARTTrainer(GRPOTrainer):
    """
    Generic trainer for DiffusionART, with the epoch loop of GRPOTrainer.

    - Each optimizer step of an epoch gets an equal part of every prompt group. If the
      sampler puts the windows inside each batch, for example 1152 images on 128 GPUs
      or one whole prompt group per batch, each step takes the samples of its
      window from every batch. A rank then trains 4 or 5 of its 9 samples, and the loss
      scale keeps the mean over all the samples of the step on all ranks, as in the
      official code.
    - The algorithm calculates the guidance of each group from the rollouts of all
      ranks, before the trainer splits the samples into optimizer steps.
    - The trainer trains on every sample. A sample of a group with equal rewards has
      no loss, but it stays in the mean, as in the official code.
    - The trainer calls ema_step only after an optimizer step.
    - A multi-reward, that is a reward with weights, trains on the weighted mean of the
      group z-scores of its calibrated components, and not on its headline score.

    The algorithm provides the GRPOTrainer interface, compute_guidance(samples) that
    returns the guidance of each sample, and combine_rewards(prompts, scores, weights)
    that returns the training reward of a multi-reward.
    """

    ema_every_batch = False
    split_groups = True

    def forward(self, data):
        results = super().forward(data)
        if "loss_scale" in data:
            results["loss"] = results["loss"] * data["loss_scale"]
        return results

    def sample_epoch(self) -> tuple[dict, dict]:
        samples, metrics = super().sample_epoch()
        samples["guidance"] = self.algorithm.compute_guidance(samples)
        return samples, metrics

    def compute_advantages(self, prompts: list[str], rewards: dict) -> np.ndarray:
        weights = getattr(self.reward, "weights", None)
        if weights is None:
            return super().compute_advantages(prompts, rewards)
        scores = {key: rewards[f"{key}_calibrated"] for key in weights}
        reward = self.algorithm.combine_rewards(prompts, scores, weights)
        return self.algorithm.compute_advantages(prompts, reward)

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def make_batches(self, samples: dict) -> list[dict]:
        num_batches, updates = self.num_batches_per_epoch, self.updates_per_epoch
        # The sampler says whether the windows live inside each batch.
        if not self.train_dataloader.batch_sampler.split_batches:
            return super().make_batches(samples)
        # The window and the batch of each sample of this rank, in the sampler order.
        windows = self.train_dataloader.batch_sampler.window_ids().repeat(num_batches)
        batch_ids = torch.arange(len(windows)) // self.sample_batch_size
        batches = []
        for w in range(updates):
            for b in range(num_batches):
                mask = ((windows == w) & (batch_ids == b)).to(self.model.device)
                batch = {k: v[mask] for k, v in samples.items()}
                # The batch mean times this scale, averaged over the ranks and the
                # batches of the step, is the mean over all the samples of the step.
                count = mask.sum().item()
                batch["loss_scale"] = count * updates / self.sample_batch_size
                batches.append(batch)
        return batches
