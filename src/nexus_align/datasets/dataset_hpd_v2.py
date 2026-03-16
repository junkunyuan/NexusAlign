"""HPDv2 dataset and benchmark for image generation."""

import os
import json

from nexus_align.core import BaseTextDataset


class HPDv2Dataset(BaseTextDataset):
    """
    HPDv2 dataset for image generation.

    References:
        - HPDv2 paper: https://arxiv.org/pdf/2306.09341.
        - Dataset: https://huggingface.co/datasets/ymhao/HPDv2.
    """

    def __init__(self, kwargs: dict) -> None:
        # Load prompts
        data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        data_path = kwargs["data"]["path"]
        data_path = os.path.join(data_and_model_dir, data_path, "train.json")
        print(f"⏳ Loading HPDv2 dataset from <{data_path}>")
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                texts.append(item["prompt"])
        
        super().__init__(
            text=texts, 
            sample_ratio=kwargs["data"]["load"]["sample_ratio"], 
            dedup=kwargs["data"]["load"]["deduplicate"], 
            remove_non_english=kwargs["data"]["load"]["remove_non_english"],
        )

        # TODO: implement cache
        self.cache_dir = kwargs["data"]["load"]["cache_dir"]
        if not isinstance(self.cache_dir, str) or not os.path.exists(self.cache_dir):
            self.cache_dir = None


class HPDv2Benchmark(BaseTextDataset):
    """
    HPDv2 Benchmark.

    Info: 3200 prompts (anime 800 + concept-art 800 + paintings 800 + photo 800)
    
    References:
        - HPDv2 paper: https://arxiv.org/pdf/2306.09341.
        - Dataset: https://huggingface.co/datasets/ymhao/HPDv2/.
    """

    def __init__(self, kwargs: dict) -> None:
        import hpsv2

        data_dict = hpsv2.benchmark_prompts("all")
        text = [prompt for v in data_dict.values() for prompt in v]

        super().__init__(
            text=text, 
            sample_ratio=kwargs["data"]["load"]["sample_ratio"],
            dedup=kwargs["data"]["load"]["deduplicate"],
            remove_non_english=kwargs["data"]["load"]["remove_non_english"],
        )
