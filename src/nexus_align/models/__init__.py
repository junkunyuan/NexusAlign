"""Model registry: register built-in model factories by name on import."""

from nexus_align.registry import registry
from nexus_align.models.meanflow_dit import MeanFlowDiTModelNames, MeanFlowDiTModel
from nexus_align.models.imeanflow_dit import IMeanFlowDiTModelNames, IMeanFlowDiTModel
from nexus_align.models.sd3 import SD3Model

for _name in MeanFlowDiTModelNames:
    registry.register("model", _name, MeanFlowDiTModel)
for _name in IMeanFlowDiTModelNames:
    registry.register("model", _name, IMeanFlowDiTModel)
registry.register("model", "sd3_5_medium", SD3Model)
