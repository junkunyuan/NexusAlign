"""Text dataset: one text per sample, with an optional metadata dict."""

import os
import json
from typing import Any

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Texts of one split, read from <path>/<file> with {split} filled in.

    - A .txt file holds one text per line, with empty metadata.
    - A .jsonl file holds one metadata dict per line, with the text under text_key.
    """

    def __init__(self, cfg_data, split: str | None = None) -> None:
        self.name = cfg_data.name
        self.split = split or cfg_data.split
        file_path = os.path.join(cfg_data.path, cfg_data.file.format(split=self.split))
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                self.metadatas = [json.loads(line) for line in f]
                self.texts = [item[cfg_data.text_key] for item in self.metadatas]
            else:
                self.texts = [line.strip() for line in f]
                self.metadatas = [{} for _ in self.texts]

        info = ["\n📚 Dataset:"]
        info += [f"    dataset: {self.name}"]
        info += [f"    split: {self.split}"]
        info += [f"    sample_count: {len(self)}"]
        print("\n".join(info))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.texts[index], "metadata": self.metadatas[index]}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> tuple[list[str], list[dict]]:
        """Collate samples into a list of texts and a list of metadata dicts."""
        return [b["text"] for b in batch], [b["metadata"] for b in batch]
