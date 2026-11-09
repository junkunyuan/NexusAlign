"""Model registry: register built-in model classes by name on import."""

from nexus_align.registry import registry
from nexus_align.models.meanflow_dit import MeanFlowDiTModelNames, MeanFlowDiTModel
from nexus_align.models.imeanflow_dit import IMeanFlowDiTModelNames, IMeanFlowDiTModel
from nexus_align.models.pmeanflow_dit import PMeanFlowDiTModelNames, PMeanFlowDiTModel
from nexus_align.models.jit_dit import JiTDiTModelNames, JiTDiTModel
from nexus_align.models.pixeldit import PixelDiTModelNames, PixelDiTModel
from nexus_align.models.pixeldit2 import PixelDiT2ModelNames, PixelDiT2Model
from nexus_align.models.sd3 import SD3Model
from nexus_align.models.twintilt_sd3 import TwinTiltSD3Model

for _name in MeanFlowDiTModelNames:
    registry.register("model", _name, MeanFlowDiTModel)
for _name in IMeanFlowDiTModelNames:
    registry.register("model", _name, IMeanFlowDiTModel)
for _name in PMeanFlowDiTModelNames:
    registry.register("model", _name, PMeanFlowDiTModel)
for _name in JiTDiTModelNames:
    registry.register("model", _name, JiTDiTModel)
for _name in PixelDiTModelNames:
    registry.register("model", _name, PixelDiTModel)
for _name in PixelDiT2ModelNames:
    registry.register("model", _name, PixelDiT2Model)
registry.register("model", "sd3_5_medium", SD3Model)
registry.register("model", "twintilt_sd3", TwinTiltSD3Model)
