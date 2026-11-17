"""
Shared group machinery of the MiniMax-H3-Image algorithms.

H3 packs the text and image rows of one prompt into one sequence, so each prompt group
lives on one rank. The algorithms share how they chunk a group, take the native PDD
schedule, look up the group of a training batch, and draw its noise. Each algorithm
keeps its own sample, rollout, and forward.

References:
    - MiniMax-H3-Image:
        - Weights: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
"""

import torch
import torch.distributed as dist


class MiniMaxH3ImageGroup:
    """Mixin for the chunks, the schedule, the layouts, and the noise of a group."""

    def init_groups(self, cfg_algorithm) -> None:
        """Read the chunk configs, and start the group list of the first epoch."""
        if self.resolution % 32:
            raise ValueError("❌ The H3-Image resolution must be a multiple of 32")
        # Members per forward pass. A group of 768-pixel rows does not fit in one
        # pass, so the rollout integrates the members in chunks, and each training
        # micro step takes one chunk at one PDD step.
        self.chunk_size = cfg_algorithm.get("chunk_size", 4)
        self.max_groups = cfg_algorithm.get("max_groups", 1024)  # groups per rank and epoch, for the noise seeds
        self.set_batch_members(self.group_size)
        self.sigmas = None  # the native PDD grid
        # The prompt embeddings and the layout of each group of this epoch, which the
        # training batches look up by their slot.
        self.groups = []

    def set_batch_members(self, members: int) -> None:
        """
        Fix the chunks of one training batch, which the trainer knows.

        A batch holds one group, or, with disjoint updates, group_size /
        updates_per_epoch members of one.
        """
        if members % self.chunk_size:
            raise ValueError(
                f"❌ chunk_size {self.chunk_size} must divide the {members} members "
                f"of a training batch"
            )
        self.num_chunks = members // self.chunk_size
        self.num_micro_steps = self.num_steps * self.num_chunks

    def bind_schedule(self, pipeline) -> None:
        """Take the native PDD schedule of the model wrapper."""
        self.sigmas = pipeline.sigmas
        if self.num_steps != len(pipeline.timesteps):
            raise ValueError(
                f"❌ H3-Image samples {len(pipeline.timesteps)} PDD steps, "
                f"not num_steps={self.num_steps}"
            )

    def group_layout(self, data: dict) -> tuple:
        """Return the entry of this batch's group: its embeddings, layout, and more."""
        slots = data["group_slots"]
        if int(slots.min()) != int(slots.max()):
            raise ValueError("❌ One H3-Image batch holds the images of one group")
        return self.groups[int(slots[0])]

    def group_seed(self, epoch: int, slot: int) -> int:
        """Return the noise seed of one group, which is unique in the run."""
        rank = (epoch * dist.get_world_size() + dist.get_rank()) * self.max_groups
        return (rank + slot) * self.group_size

    def group_noise(self, seed: int, latent_size: int, num_images: int) -> torch.Tensor:
        """Draw the initial rows of one group, one own noise per member."""
        noise = [
            self.pipeline.initial_noise(seed + i, latent_size)
            for i in range(num_images)
        ]
        return torch.stack(noise).float()
