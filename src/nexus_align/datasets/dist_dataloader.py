"""Dataloader construction: distributed dataloader builder."""

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
    Batch i of epoch e is seeded by seed + e * num_batches + i, so all ranks agree.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        num_batches: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_batches = num_batches
        self.seed = seed
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        total = self.world_size * batch_size
        assert total % group_size == 0, (
            f"❌ group_size {group_size} must divide world_size * batch_size {total}"
        )
        self.num_groups = total // group_size
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * self.num_batches + i)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.num_groups]
            repeated = [idx for idx in indices.tolist() for _ in range(self.group_size)]
            order = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[k] for k in order]
            start = self.rank * self.batch_size
            yield shuffled[start: start + self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 0,
    sampler: Sampler | None = None,
    num_workers: int = 4,
    drop_last: bool = False,
    grad_accu_step: int = 1,
    generator: torch.Generator = None
) -> DataLoader:
    """
    Build a distributed dataloader and print a data summary.

    trainer provides *_batch_size, shuffle/drop_last, and grad_accu_step.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Calculate batchsize.
    batch_size_per_rank = batch_size // world_size // grad_accu_step
    batch_size = batch_size_per_rank * world_size * grad_accu_step

    # Build data loader.
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

    # Print info.
    info = ["\n📚 DataLoader:"]
    info += [f"    batch_size_per_rank: {batch_size_per_rank}"]
    info += [f"    effective_batch_size: {batch_size}"]
    info += [f"    batch_count: {len(dataloader)}"]
    info += [f"    num_workers: {num_workers}"]
    if drop_last:
        info += ["    drop_last: true"]
    print("\n".join(info))

    return dataloader
