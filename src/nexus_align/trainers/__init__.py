"""Trainer registry: register built-in trainers by name on import."""

from nexus_align.registry import registry
from nexus_align.trainers.sft_trainer import SFTTrainer
from nexus_align.trainers.grpo_trainer import GRPOTrainer
from nexus_align.trainers.nft_trainer import NFTTrainer

registry.register("trainer", "sft", SFTTrainer)
registry.register("trainer", "grpo", GRPOTrainer)
registry.register("trainer", "nft", NFTTrainer)
