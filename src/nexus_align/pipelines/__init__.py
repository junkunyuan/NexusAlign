from nexus_align.core.registry import registry
from nexus_align.pipelines.pipeline_flux import (
    FluxTrainPipeline,
    FluxInferPipeline,
)
from nexus_align.pipelines.pipeline_qwen_image import (
    QwenImageTrainPipeline,
    QwenImageInferPipeline,
)
from nexus_align.pipelines.pipeline_sd3 import (
    SD3TrainPipeline,
    SD3InferPipeline,
)
from nexus_align.pipelines.pipeline_z_image import (
    ZImageTrainPipeline,
    ZImageInferPipeline,
)

registry.register("pipeline", "flux_train", FluxTrainPipeline)
registry.register("pipeline", "flux_infer", FluxInferPipeline)
registry.register("pipeline", "qwen_image_train", QwenImageTrainPipeline)
registry.register("pipeline", "qwen_image_infer", QwenImageInferPipeline)
registry.register("pipeline", "sd3_train", SD3TrainPipeline)
registry.register("pipeline", "sd3_infer", SD3InferPipeline)
registry.register("pipeline", "z_image_train", ZImageTrainPipeline)
registry.register("pipeline", "z_image_infer", ZImageInferPipeline)

__all__ = [
    "FluxTrainPipeline",
    "FluxInferPipeline",
    "QwenImageTrainPipeline",
    "QwenImageInferPipeline",
    "SD3TrainPipeline",
    "SD3InferPipeline",
    "ZImageTrainPipeline",
    "ZImageInferPipeline",
]
