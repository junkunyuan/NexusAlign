from nexus_align.core.registry import registry
from nexus_align.models.model_flux import FluxModel
from nexus_align.models.model_qwen_image import QwenImageModel
from nexus_align.models.model_sd3 import SD3Model
from nexus_align.models.model_z_image import ZImageModel

registry.register("model", "flux", FluxModel)
registry.register("model", "qwen_image", QwenImageModel)
registry.register("model", "sd3", SD3Model)
registry.register("model", "z_image", ZImageModel)
