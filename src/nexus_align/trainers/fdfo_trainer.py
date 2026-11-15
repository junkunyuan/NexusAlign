"""
FDFO trainer: online RL epochs of trajectory pairs, with optimizer steps on random parts
of the (sample, timestep) slots.

License: this file adapts the official code, Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES, under the NVIDIA Source Code License v1 (Non-Commercial). The license is in
reproduce/fdfo/LICENSE.txt, and it limits this file to non-commercial use.

References:
    - FDFO:
        - Official code: https://github.com/NVlabs/finite-difference-flow-optimization
            - fdfo/train_loop.py: the slot shuffle, the micro-batches, and the gradient
              accumulation of train_fdfo
            - train.py: configure_training_batches
"""

from collections import defaultdict
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset

from nexus_align.trainers.grpo_trainer import GRPOTrainer
from nexus_align.datasets.dist_dataloader import DistributedGroupSampler


class FDFOTrainer(GRPOTrainer):
    """
    Online RL trainer for FDFO, with the rollouts and the rewards of GRPOTrainer.

    - Each rank gets the two images of a pair in a row, so that they can share one
      noise. sample_batch_size counts the images, so it must be even.
    - One trainer step is one epoch, so updates_per_epoch must be 1. The epoch makes
      gradient_steps_per_epoch optimizer steps. Each step takes a random equal part of
      the (sample, timestep) slots of this rank, in forward passes of at most
      max_train_batch_size slots.
    """

    ema_every_batch = False

    def __init__(self, cfg_trainer, *args, **kwargs) -> None:
        if cfg_trainer["sample_batch_size"] % 2:
            raise ValueError("❌ sample_batch_size counts pair images; make it even")
        if cfg_trainer["updates_per_epoch"] != 1:
            raise ValueError("❌ FDFO steps once per epoch; set updates_per_epoch: 1")
        super().__init__(cfg_trainer, *args, **kwargs)
        print(f"    gradient steps per trainer step: {self.gradient_steps_per_epoch}")

    def get_train_dataloader(self, dataset: Dataset) -> DataLoader:
        """Build a dataloader that repeats each prompt twice in a row on one rank."""
        sampler = DistributedGroupSampler(
            dataset,
            batch_size=self.sample_batch_size // 2,
            group_size=1,
            num_batches=self.num_batches_per_epoch,
            seed=self.seed,
        )

        def collate_fn(batch):
            return dataset.collate_fn([item for item in batch for _ in range(2)])

        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def train_epoch(self, samples: dict, rollout_metrics: dict) -> None:
        """Train on the slots of this rank in gradient_steps_per_epoch steps."""
        num_samples, num_steps = samples["xs"].shape[:2]
        num_slots = num_samples * num_steps
        if num_slots % self.gradient_steps_per_epoch:
            raise ValueError(
                f"❌ The {num_slots} slots of a rank must split into "
                f"gradient_steps_per_epoch ({self.gradient_steps_per_epoch}) parts"
            )
        part_size = num_slots // self.gradient_steps_per_epoch
        batch_size = max(
            b for b in range(1, self.max_train_batch_size + 1) if part_size % b == 0
        )
        rollout_metrics["actual_batch_size"] = batch_size

        self.train_mode()
        slots = torch.randperm(num_slots, device=self.model.device)
        info = defaultdict(list)
        for part in slots.split(part_size):
            self.zero_grad()
            batches = part.split(batch_size)
            for i, batch in enumerate(batches):
                data = {k: v[batch // num_steps] for k, v in samples.items()}
                data["step"] = batch % num_steps
                with nullcontext() if i == len(batches) - 1 else self.no_sync():
                    results = self.forward(data)
                    self.backward(results["loss"] / len(batches))
                for key, value in results["metrics"].items():
                    info[key].append(value)
            self.clip_grad()
            self.optimizer_step()
            self.lr_scheduler_step()
        self.step += 1
        self.total_step += 1
        self.end_step(info, rollout_metrics)
