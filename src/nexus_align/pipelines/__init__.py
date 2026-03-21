from nexus_align.core.registry import registry
from nexus_align.pipelines.pipeline_flux import (
    FluxTrainPipeline,
    FluxInferPipeline,
)
from nexus_align.pipelines.pipeline_flux_dpo import FluxDPOTrainPipeline

registry.register("pipeline", "flux_train", FluxTrainPipeline)
registry.register("pipeline", "flux_dpo_train", FluxDPOTrainPipeline)
registry.register("pipeline", "flux_infer", FluxInferPipeline)

__all__ = [
    "FluxTrainPipeline",
    "FluxDPOTrainPipeline",
    "FluxInferPipeline",
]
