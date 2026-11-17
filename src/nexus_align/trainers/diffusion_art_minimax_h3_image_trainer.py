"""
DiffusionART trainer for MiniMax-H3-Image: whole prompt groups, two disjoint updates.

References:
    - DiffusionART:
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/backends/sd3.py: run and _art_update, for the two disjoint
              updates per rollout
"""

from nexus_align.trainers.art_trainer import ARTTrainer


class DiffusionARTMiniMaxH3ImageTrainer(ARTTrainer):
    """
    ART trainer with the epoch loop of ARTTrainer, on whole prompt groups.

    A packed H3 sequence holds one prompt, so each rank rolls out whole groups and one
    rollout batch is one group. The sampler then splits the members of every group into
    the windows of the updates_per_epoch optimizer steps, so each step still trains on
    an equal part of every group, as the official code does.
    """

    whole_groups = True  # the sampler gives each rank whole prompt groups

    def __init__(self, cfg_trainer, *args, **kwargs) -> None:
        # One training batch holds the members of one window of one group. The
        # algorithm needs its chunks before the trainer prints its plan.
        algorithm = kwargs["algorithm"]
        algorithm.set_batch_members(
            algorithm.group_size // cfg_trainer["updates_per_epoch"]
        )
        super().__init__(cfg_trainer, *args, **kwargs)
        if self.sample_batch_size != self.group_size:
            raise ValueError(
                f"❌ One rollout batch must be one group: sample_batch_size "
                f"({self.sample_batch_size}) must be group_size ({self.group_size})"
            )

    def sample_epoch(self) -> tuple[dict, dict]:
        """Drop the groups of the last epoch, and roll out the groups of this one."""
        self.algorithm.groups = []
        return super().sample_epoch()
