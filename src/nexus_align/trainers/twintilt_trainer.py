"""
TwinTilt trainer: online RL epochs that solve the targets of each group over all ranks.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private, twintilt-twintilt.zip)
            - src/nexus_align/trainers/twintilt_trainer.py:
              TwinTiltTrainer.sample_epoch, train_update; RolloutEMA.update
"""

from collections import defaultdict

import numpy as np
import torch

from nexus_align.engine.distributed import gather_tensor
from nexus_align.trainers.grpo_trainer import GRPOTrainer


class TwinTiltTrainer(GRPOTrainer):
    """
    Online RL trainer for TwinTilt, with the epoch loop and the dataloader of
    GRPOTrainer.

    - The images of a group can be on several ranks. Each group has an ID that is the
      same on all ranks, and the algorithm solves the targets of each group from the
      rollouts of all ranks.
    - The trainer trains on every sample, at every sampling timestep.
    - The EMA samples the rollouts. The trainer updates it after each optimizer step
      with the decay min(ema_warmup_rate * total_step, ema_decay).

    The algorithm provides the GRPOTrainer interface, but rollout(prompts, group_ids)
    takes the group IDs, and build_targets(samples, scores, reward) takes the scores of
    this rank and returns the samples with their targets and the metrics.
    """

    ema_every_batch = False

    def ema_step(self):
        """Update the EMA with decay min(ema_warmup_rate * total_step, ema_decay)."""
        if self.model.ema is None:
            return
        decay = min(self.ema_warmup_rate * self.total_step, self.ema_decay)
        self.model.ema_step(decay=decay)

    @torch.no_grad()
    def sample_epoch(self) -> tuple[dict, dict]:
        """Sample one epoch of response groups, reward them, and solve their targets."""
        # 1. Roll out each batch and submit its rewards asynchronously.
        self.model.model.eval()
        sampler = self.train_dataloader.batch_sampler
        sampler.set_epoch(self.epoch)
        num_prompts = len(self.train_dataset)
        batches, futures = [], []
        # The sampler gives the same batches in each pass, so a second pass gives the
        # dataset indices. A batch never draws one prompt for two groups.
        for i, ((prompts, metadatas), indices) in enumerate(
            zip(self.train_dataloader, sampler)
        ):
            # Each group ID is unique in the run, and it is the same on all ranks.
            batch = self.epoch * self.num_batches_per_epoch + i
            group_ids = [batch * num_prompts + j for j in indices]
            out = self.algorithm.rollout(prompts, group_ids)
            future = self.executor.submit(
                self.reward, out["responses"], prompts, metadatas, True
            )
            futures.append(future)
            batches.append(out["samples"])
        samples = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}

        # 2. Get the advantages and the targets of the groups of all ranks.
        scores = defaultdict(list)
        for future in futures:
            for key, value in future.result().items():
                scores[key] += list(value)
        device = self.model.device
        scores = {
            key: torch.as_tensor(value, device=device).float()
            for key, value in scores.items()
        }
        samples, metrics = self.algorithm.build_targets(samples, scores, self.reward)

        # 3. Get the main reward metrics of all ranks over the samples that they apply
        # to. A value of -10 marks the other samples.
        for key in self.reward.metrics:
            if key in scores:
                value = gather_tensor(scores[key]).cpu().numpy()
                metrics[f"reward_{key}"] = float(np.mean(value[value != -10]))
        return samples, metrics

    def filter_samples(self, samples: dict) -> dict:
        return samples
