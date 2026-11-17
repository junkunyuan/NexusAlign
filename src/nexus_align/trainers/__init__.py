"""Trainer registry: register built-in trainers by name on import."""

from nexus_align.registry import registry
from nexus_align.trainers.sft_trainer import SFTTrainer
from nexus_align.trainers.grpo_trainer import GRPOTrainer
from nexus_align.trainers.nft_trainer import NFTTrainer
from nexus_align.trainers.awm_minimax_h3_image_trainer import AWMMiniMaxH3ImageTrainer
from nexus_align.trainers.diffusion_nft_minimax_h3_image_trainer import (
    DiffusionNFTMiniMaxH3ImageTrainer,
)
from nexus_align.trainers.refl_trainer import ReFLTrainer
from nexus_align.trainers.art_trainer import ARTTrainer
from nexus_align.trainers.diffusion_art_minimax_h3_image_trainer import (
    DiffusionARTMiniMaxH3ImageTrainer,
)
from nexus_align.trainers.awm_trainer import AWMTrainer
from nexus_align.trainers.fdfo_trainer import FDFOTrainer
from nexus_align.trainers.twintilt_trainer import TwinTiltTrainer
from nexus_align.trainers.twintilt_minimax_h3_image_trainer import (
    TwinTiltMiniMaxH3ImageTrainer
)


registry.register("trainer", "sft", SFTTrainer)
registry.register("trainer", "grpo", GRPOTrainer)
registry.register("trainer", "nft", NFTTrainer)
registry.register(
    "trainer", "awm_minimax_h3_image", AWMMiniMaxH3ImageTrainer
)
registry.register(
    "trainer", "diffusion_nft_minimax_h3_image", DiffusionNFTMiniMaxH3ImageTrainer
)
registry.register("trainer", "refl", ReFLTrainer)
registry.register("trainer", "art", ARTTrainer)
registry.register(
    "trainer", "diffusion_art_minimax_h3_image", DiffusionARTMiniMaxH3ImageTrainer
)
registry.register("trainer", "awm", AWMTrainer)
registry.register("trainer", "fdfo", FDFOTrainer)
registry.register("trainer", "twintilt", TwinTiltTrainer)
registry.register(
    "trainer", "twintilt_minimax_h3_image", TwinTiltMiniMaxH3ImageTrainer
)
