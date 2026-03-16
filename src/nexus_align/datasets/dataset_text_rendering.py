"""Text rendering dataset and benchmark for image generation."""

import os
import csv

from nexus_align.core import BaseTextDataset


class TextRenderingDataset(BaseTextDataset):
    """
    Text rendering dataset for image generation.

    CSV columns: index, text, prompt, class, text_length, prompt_length.
    The ``prompt`` column is the full generation prompt.
    """

    def __init__(self, kwargs: dict) -> None:
        data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        data_path = kwargs["data"]["path"]
        data_path = os.path.join(data_and_model_dir, data_path, "train_set.csv")
        print(f"⏳ Loading TextRendering dataset from <{data_path}>")

        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["prompt"])

        super().__init__(
            text=texts,
            sample_ratio=kwargs["data"]["load"]["sample_ratio"],
            dedup=kwargs["data"]["load"]["deduplicate"],
            remove_non_english=kwargs["data"]["load"]["remove_non_english"],
        )

        self.cache_dir = kwargs["data"]["load"]["cache_dir"]
        if not isinstance(self.cache_dir, str) or not os.path.exists(self.cache_dir):
            self.cache_dir = None


class TextRenderingBenchmark(BaseTextDataset):
    """
    Text rendering benchmark.

    CSV columns: index, text, prompt, class, text_length, prompt_length, position.
    The `prompt` column is the full generation prompt.
    """

    def __init__(self, kwargs: dict) -> None:
        data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        data_path = kwargs["data"]["path"]
        data_path = os.path.join(data_and_model_dir, data_path, "test_set.csv")
        print(f"⏳ Loading TextRendering benchmark from <{data_path}>")

        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["prompt"])

        super().__init__(
            text=texts,
            sample_ratio=kwargs["data"]["load"]["sample_ratio"],
            dedup=kwargs["data"]["load"]["deduplicate"],
            remove_non_english=kwargs["data"]["load"]["remove_non_english"],
        )
