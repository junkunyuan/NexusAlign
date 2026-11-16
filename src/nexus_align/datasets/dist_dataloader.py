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

    If whole_groups is True, the batch keeps the group order, so each rank gets whole
    groups. Then batch_size must be a multiple of group_size.

    If num_windows > 1, each window gets group_size / num_windows copies of every
    group, as follows:
    - If whole_groups is True, every batch holds whole groups, and the windows split
      the samples of each batch. Then batch_size must also be a multiple of
      num_windows.
    - Else if num_windows divides num_batches, the batches make num_windows equal
      windows. Batch i of each window holds the copies of the same prompts.
    - Otherwise, every batch holds whole groups, and the windows alternate along its
      global list. Then each rank gets batch_size / num_windows samples of each window,
      rounded up or down, and window_ids gives the window of each of its samples.
    In the first and the last case, split_batches is True and window_ids gives the
    window of each sample of a batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        num_batches: int,
        seed: int = 0,
        num_windows: int = 1,
        whole_groups: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_batches = num_batches
        self.seed = seed
        self.num_windows = num_windows
        self.whole_groups = whole_groups
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Windows inside each batch, which window_ids then gives.
        self.split_batches = whole_groups or num_batches % num_windows != 0
        total = self.world_size * batch_size
        total = total if self.split_batches else total * num_windows
        assert total % group_size == 0 and group_size % num_windows == 0, (
            f"❌ group_size {group_size} must divide the samples of a prompt draw "
            f"{total}, and num_windows {num_windows} must divide group_size"
        )
        assert not (whole_groups and batch_size % group_size), (
            f"❌ whole_groups needs batch_size {batch_size} to be a multiple of "
            f"group_size {group_size}"
        )
        assert not (whole_groups and batch_size % num_windows), (
            f"❌ whole_groups needs batch_size {batch_size} to be a multiple of "
            f"num_windows {num_windows}, which split the samples of a batch"
        )
        assert batch_size >= num_windows, (
            f"❌ batch_size {batch_size} must be at least num_windows {num_windows}"
        )
        self.num_groups = total // group_size
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        if self.split_batches:
            yield from self.iter_split_batches()
            return
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
                # Whole groups keep the group order instead.
                if self.whole_groups:
                    shuffled = repeated
                else:
                    order = torch.randperm(len(repeated), generator=g).tolist()
                    shuffled = [repeated[k] for k in order]

                # Keep the samples of this rank.
                start = self.rank * self.batch_size  # this rank's slice
                batches[w * per_window + i] = shuffled[start: start + self.batch_size]
        yield from batches  # batch_size prompt indices per batch

    def iter_split_batches(self):
        """Yield batches of whole groups, in which the windows alternate."""
        # With whole groups, a batch holds whole groups and the windows split it.
        copies = self.group_size
        if not self.whole_groups:
            copies = self.group_size // self.num_windows
        for i in range(self.num_batches):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * self.num_batches + i)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.num_groups]

            # Shuffle the copies of each window, then alternate the windows, for
            # example ["dog", "sun", "sun", "dog"] for windows 0, 1, 0, 1.
            if self.whole_groups:
                shuffled = [idx for idx in indices.tolist() for _ in range(copies)]
            else:
                windows = []
                for _ in range(self.num_windows):
                    repeated = [idx for idx in indices.tolist() for _ in range(copies)]
                    order = torch.randperm(len(repeated), generator=g).tolist()
                    windows.append([repeated[k] for k in order])
                shuffled = [idx for column in zip(*windows) for idx in column]

            start = self.rank * self.batch_size  # this rank's slice
            yield shuffled[start: start + self.batch_size]

    def window_ids(self) -> torch.Tensor:
        """Return the window of each sample of this rank in a batch of split windows."""
        start = self.rank * self.batch_size
        return torch.arange(start, start + self.batch_size) % self.num_windows

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
