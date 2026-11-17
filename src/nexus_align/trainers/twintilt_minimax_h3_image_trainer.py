"""
TwinTilt trainer for MiniMax-H3-Image: whole prompt groups, one batch per group.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private)
            - src/nexus_align/twintilt_h3/trainer.py: TwinTiltH3Trainer.sample_epoch,
              train_update
"""

from nexus_align.trainers.twintilt_trainer import TwinTiltTrainer
from nexus_align.trainers.minimax_h3_image_group_trainer import (
    MiniMaxH3ImageGroupTrainer,
)


class TwinTiltMiniMaxH3ImageTrainer(MiniMaxH3ImageGroupTrainer, TwinTiltTrainer):
    """TwinTilt trainer with the epoch loop of TwinTiltTrainer, on whole groups."""
