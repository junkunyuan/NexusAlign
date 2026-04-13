"""Text rendering dataset and benchmark for image generation."""

import os
import json

from nexus_align.core import BaseTextDataset


class TextRenderingDataset(BaseTextDataset):
    """
    Text rendering dataset for image generation.

    JSONL fields: index, text, prompt, class, text_length, prompt_length.
    The ``prompt`` field is the full generation prompt.
    """

    def __init__(self, kwargs: dict) -> None:
        data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        data_path = kwargs["data"]["path"]
        train_file = kwargs["data"]["load"].get("train_file", "train_set.jsonl")
        data_path = os.path.join(data_and_model_dir, data_path, train_file)
        print(f"⏳ Loading TextRendering dataset from <{data_path}>")

        texts = []
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
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

    JSONL fields: index, text, prompt, class, text_length, prompt_length, position.
    The ``prompt`` field is the full generation prompt.
    """

    def __init__(self, kwargs: dict) -> None:
        data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        data_path = kwargs["data"]["path"]
        test_file = kwargs["data"]["load"].get("test_file", "test_set.jsonl")
        data_path = os.path.join(data_and_model_dir, data_path, test_file)
        print(f"⏳ Loading TextRendering benchmark from <{data_path}>")

        texts = []
        with open(data_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    texts.append(row["prompt"])

        super().__init__(
            text=texts,
            sample_ratio=kwargs["data"]["load"]["sample_ratio"],
            dedup=kwargs["data"]["load"]["deduplicate"],
            remove_non_english=kwargs["data"]["load"]["remove_non_english"],
        )
