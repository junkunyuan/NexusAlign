"""
AWM trainer for MiniMax-H3-Image: whole prompt groups, one batch per group.

References:
    - AWM:
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py: the epoch loop of
              main, with one optimizer step per epoch
"""

from nexus_align.trainers.awm_trainer import AWMTrainer
from nexus_align.trainers.minimax_h3_image_group_trainer import (
    MiniMaxH3ImageGroupTrainer,
)


class AWMMiniMaxH3ImageTrainer(MiniMaxH3ImageGroupTrainer, AWMTrainer):
    """AWM trainer with the epoch loop of AWMTrainer, on whole prompt groups."""
