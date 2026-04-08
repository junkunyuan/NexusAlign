from nexus_align.core.registry import registry
from nexus_align.algorithms.grpo import GRPOAlgorithm

registry.register("algorithm", "grpo", GRPOAlgorithm)


__all__ = ["GRPOAlgorithm"]
