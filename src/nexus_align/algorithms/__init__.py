"""Algorithm registry: register built-in algorithms by name on import."""

from nexus_align.registry import registry
from nexus_align.algorithms.meanflow import MeanFlow
from nexus_align.algorithms.imeanflow import IMeanFlow
from nexus_align.algorithms.flow_grpo import FlowGRPO
from nexus_align.algorithms.diffusion_nft import DiffusionNFT

registry.register("algorithm", "meanflow", MeanFlow)
registry.register("algorithm", "imeanflow", IMeanFlow)
registry.register("algorithm", "flow_grpo", FlowGRPO)
registry.register("algorithm", "diffusion_nft", DiffusionNFT)
