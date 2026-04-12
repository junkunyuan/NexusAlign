from nexus_align.core.registry import registry
from nexus_align.algorithms.grpo import GRPOAlgorithm
from nexus_align.algorithms.dpo import DPOAlgorithm

registry.register("algorithm", "grpo", GRPOAlgorithm)
registry.register("algorithm", "dpo", DPOAlgorithm)


__all__ = ["GRPOAlgorithm", "DPOAlgorithm"]
