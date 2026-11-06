"""
Dataloader construction: distributed dataloader builder.

References:
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
            - scripts/train_sd3.py: DistributedKRepeatSampler
"""

import random

import numpy as np

import torch
import torch.distributed as dist

from torch.utils.data import Dataset, Sampler, DataLoader


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedGroupSampler(Sampler):
    """
    Batch sampler that repeats each sampled prompt group_size times across all ranks.

    Every batch draws world_size * batch_size / group_size prompts, repeats each one
    group_size times, shuffles them, and gives each rank its slice of batch_size.
    The seed of batch i in epoch e is seed + e * num_batches + i, so all ranks agree.

    If num_windows > 1, the batches make num_windows equal windows. Batch i of each
    window holds group_size / num_windows copies of the same prompts, so each window
    gets an equal part of every group.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        num_batches: int,
        seed: int = 0,
        num_windows: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_batches = num_batches
        self.seed = seed
        self.num_windows = num_windows
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        total = self.world_size * batch_size * num_windows
        assert total % group_size == 0 and group_size % num_windows == 0, (
            f"❌ group_size {group_size} must divide world_size * batch_size * "
            f"num_windows {total}, and num_windows {num_windows} must divide group_size"
        )
        assert num_batches % num_windows == 0, (
            f"❌ num_windows {num_windows} must divide num_batches {num_batches}"
        )
        self.num_groups = total // group_size
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        per_window = self.num_batches // self.num_windows
        copies = self.group_size // self.num_windows
        batches = [None] * self.num_batches
        for i in range(per_window):
            # Make sure the sample order is the same across ranks.
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * self.num_batches + i)

            # Pick the first num_groups samples, for example ["dog", "sun", "car"].
            indices = torch.randperm(len(self.dataset), generator=g)[:self.num_groups]

            for w in range(self.num_windows):
                # Repeat the samples to make groups, for example
                # ["dog", "dog", "sun", "sun", "car", "car"].
                repeated = [idx for idx in indices.tolist() for _ in range(copies)]

                # Shuffle the samples into a global list, the same on all ranks.
                order = torch.randperm(len(repeated), generator=g).tolist()
                shuffled = [repeated[k] for k in order]

                # Keep the samples of this rank.
                start = self.rank * self.batch_size  # this rank's slice
                batches[w * per_window + i] = shuffled[start: start + self.batch_size]
        yield from batches  # batch_size prompt indices per batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler,
    num_workers: int,
    drop_last: bool,
    grad_accu_step: int,
    generator: torch.Generator,
) -> DataLoader:
    """
    Build a distributed dataloader and print a data summary.

    The trainer gives batch_size, sampler, drop_last, and grad_accu_step.
    """
    world_size = dist.get_world_size()

    # Calculate the batch size.
    batch_size_per_rank = batch_size // world_size // grad_accu_step
    batch_size = batch_size_per_rank * world_size * grad_accu_step

    # Build the dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_rank,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator
    )

    # Print the data summary.
    info = ["\n📚 DataLoader:"]
    info += [f"    batch_size_per_rank: {batch_size_per_rank}"]
    info += [f"    effective_batch_size: {batch_size}"]
    info += [f"    batch_count: {len(dataloader)}"]
    info += [f"    num_workers: {num_workers}"]
    if drop_last:
        info += ["    drop_last: true"]
    print("\n".join(info))

    return dataloader
