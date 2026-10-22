"""Base dataset: generic dataset."""

from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from nexus_align.datasets.utils import sample_items


class BaseTextImageDataset(Dataset, ABC):
    """
    Abstract base for image + text + label datasets.

    - Subclasses implement get_raw() to return one raw sample.
    - build_indices() samples by sample_ratio.
    - __getitem__ applies image/text transforms and returns a sample dict.
    - collate_fn batches tensor fields and keeps non-tensor fields as lists.
    """

    def __init__(
        self,
        name: str | None = None,
        cache_dir: str | None = None,
        image_transform: Callable | None = None,
        text_transform: Callable | None = None,
        sample_ratio: float | int | None = None,
        sample_seed: int | None = None,
    ) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.sample_ratio = sample_ratio
        self.sample_seed = sample_seed

        # Built by build_indices; maps dataset indices to raw sample indices.
        self._indices = None

    @abstractmethod
    def get_raw(self, index: int) -> dict[str, Any]:
        """Return a raw sample dict with keys of "image", "text", "label"."""
        ...

    def build_indices(self, num_items: int) -> None:
        """Build the active index by sampling."""
        indices = list(range(num_items))
        self._indices = sample_items(indices, self.sample_ratio, seed=self.sample_seed)
        self.print_data_info()

    def print_data_info(self) -> None:
        """Print dataset information."""
        info = ["\n📚 Dataset:"]
        info += [f"    dataset: {self.name}"]
        info += [f"    sample_count: {len(self)}"]
        if self.sample_ratio != 1.:
            info += [f"    sample_ratio: {self.sample_ratio}"]
        if self.cache_dir:
            info += [f"    cache_dir: {self.cache_dir}"]
        print("\n".join(info))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Get image, text, and label.
        raw = self.get_raw(self._indices[index])
        image, text, label = raw.get("image"), raw.get("text"), raw.get("label")

        # Data transforms.
        if image is not None and self.image_transform is not None:
            image = self.image_transform(image)
        if text is not None and self.text_transform is not None:
            text = self.text_transform(text)

        return {"image": image, "text": text, "label": label}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:  # TODO: not used
        """Collate sample dicts: stack tensor images/labels, list the rest."""
        def stack(key: str) -> Any:
            vals = [b[key] for b in batch]
            if vals and torch.is_tensor(vals[0]):
                return torch.stack(vals)
            if key == "label" and all(v is not None for v in vals):
                return torch.tensor(vals)
            return vals

        return {
            "image": stack("image"),
            "text": [b["text"] for b in batch],
            "label": stack("label"),
        }
