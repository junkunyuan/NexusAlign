"""
DiffusionNFT trainer for MiniMax-H3-Image: whole prompt groups, one batch per group.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
            - scripts/train_nft_sd3.py: the epoch loop of main
"""

from nexus_align.trainers.nft_trainer import NFTTrainer
from nexus_align.trainers.minimax_h3_image_group_trainer import (
    MiniMaxH3ImageGroupTrainer,
)


class DiffusionNFTMiniMaxH3ImageTrainer(MiniMaxH3ImageGroupTrainer, NFTTrainer):
    """NFT trainer with the epoch loop of NFTTrainer, on whole prompt groups."""
