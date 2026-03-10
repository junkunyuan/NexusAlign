from nexus_align.core.registry import registry
from nexus_align.datasets.dataset_hpd_v2 import HPDv2Dataset, HPDv2Benchmark

registry.register("dataset", "hpd_v2", HPDv2Dataset)
registry.register("dataset", "hpd_v2_benchmark", HPDv2Benchmark)
