"""Dataset registry: register built-in datasets by name on import."""

from nexus_align.registry import registry
from nexus_align.datasets.imagenet_1k import ImageNet1K
from nexus_align.datasets.text_dataset import TextDataset

registry.register("dataset", "imagenet_1k", ImageNet1K)
for _name in ("geneval", "ocr", "pickscore", "drawbench"):
    registry.register("dataset", _name, TextDataset)
