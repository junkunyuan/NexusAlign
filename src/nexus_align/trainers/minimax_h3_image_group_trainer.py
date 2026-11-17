"""
Shared epoch loop of the MiniMax-H3-Image trainers that train on whole groups.

A packed H3 sequence holds one prompt, so each rank rolls out whole groups and every
training batch is one group. One optimizer step per epoch then accumulates the
gradients of all groups and all PDD steps of all chunks.
"""


class MiniMaxH3ImageGroupTrainer:
    """Mixin that keeps whole prompt groups in the rollout and training batches."""

    whole_groups = True  # the sampler gives each rank whole prompt groups

    def __init__(self, cfg_trainer, *args, **kwargs) -> None:
        super().__init__(cfg_trainer, *args, **kwargs)
        if self.sample_batch_size != self.group_size:
            raise ValueError(
                f"❌ One batch must be one group: sample_batch_size "
                f"({self.sample_batch_size}) must be group_size ({self.group_size})"
            )
        if self.shuffle:
            raise ValueError("❌ H3-Image needs shuffle: false, to keep whole groups")

    def sample_epoch(self) -> tuple[dict, dict]:
        """Drop the groups of the last epoch, and roll out the groups of this one."""
        self.algorithm.groups = []
        return super().sample_epoch()
