from nexus_align.core.registry import registry
from nexus_align.models.model_flux import FluxModel

registry.register("model", "flux", FluxModel)
