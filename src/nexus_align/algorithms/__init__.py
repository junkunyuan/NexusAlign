"""Algorithm registry: register built-in algorithms by name on import."""

from nexus_align.registry import registry

# Visual generation post-training.
from nexus_align.algorithms.diffusion_nft import DiffusionNFT
from nexus_align.algorithms.flow_grpo import FlowGRPO
from nexus_align.algorithms.refl import ReFL

# Visual generation paradigms.
from nexus_align.algorithms.imeanflow import IMeanFlow
from nexus_align.algorithms.meanflow import MeanFlow


# Visual generation post-training.
registry.register("algorithm", "diffusion_nft", DiffusionNFT)
registry.register("algorithm", "flow_grpo", FlowGRPO)
registry.register("algorithm", "refl", ReFL)

# Visual generation paradigms.
registry.register("algorithm", "imeanflow", IMeanFlow)
registry.register("algorithm", "meanflow", MeanFlow)
