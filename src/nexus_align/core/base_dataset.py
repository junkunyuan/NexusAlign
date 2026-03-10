"""Base dataset classes."""

import os
import json
import random
import hashlib
from collections.abc import Callable
from typing import TypeVar, Hashable

from string import ascii_letters, digits, punctuation, whitespace

from PIL import Image
import torch
from torch.utils.data import Dataset


T = TypeVar("T")


def is_pure_english(text: str) -> bool:
    """Check if the text is in pure English."""
    allowed = set(ascii_letters + digits + punctuation + whitespace)
    return all(c in allowed for c in text)


def deduplicate(
    items: list[T],
    *,
    key: Callable[[T], Hashable] | str | None = None,
) -> list[T]:
    """
    Deduplicate items with flexible key extraction.

    Args:
        items: List of items to deduplicate.
        key: Callable to extract hashable key for dedup (e.g., lambda x: x["text"]).
             If str, use item[key] for dict-like items (supports nested keys via dot, e.g. "a.b").
             If None, use item itself (must be hashable).

    Returns:
        Deduplicated list (first occurrence kept).

    Examples:
        >>> deduplicate(["a", "b", "a"], key=None)
        ['a', 'b']
        >>> deduplicate([{"t": "x"}, {"t": "y"}, {"t": "x"}], key="t")
        [{'t': 'x'}, {'t': 'y'}]
    """
    if not items:
        return []

    # Resolve key function
    if key is None:
        key_fn: Callable[[T], Hashable] = lambda x: x
    elif callable(key):
        key_fn = key
    elif isinstance(key, str):
        def _get_by_key(obj, k: str):
            for part in k.split("."):
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            return obj
        key_fn = lambda x: _get_by_key(x, key)
    else:
        raise TypeError(f"key must be None, callable, or str, got {type(key)}")

    # Deduplicate by key, keeping first occurrence
    seen: set[Hashable] = set()
    result: list[T] = []
    for item in items:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            result.append(item)

    return result


def sample_items(
    items: list[T],
    sample_ratio: float | int | None = None,
) -> list[T]:
    """
    Sample a subset of items.

    Args:
        items: List of items to sample from.
        sample_ratio: 
            If <= 0, return all items unchanged.
            If < 1.1., treat as fraction (e.g. 0.5 = 50%, 1.0 = 100%).
            Else, treat as absolute count.
                      

    Returns:
        Sampled list.
    """
    n = len(items)
    if not isinstance(sample_ratio, (float, int)) or sample_ratio <= 0 or n == 0:
        return items

    if sample_ratio < 1.1:
        sample_ratio = n * sample_ratio
    
    k = min(int(sample_ratio), n)

    return random.sample(items, k)


def texts_to_md5s(texts: list[str], dedup: bool = True) -> list[str]:
    """
    Convert text to MD5 hash.

    If dedup is True, the texts have already been deduplicated.
    Else, the texts will be deduplicated by adding a suffix before generating MD5s.
    """
    if dedup:
        return [hashlib.md5(text.encode("utf-8")).hexdigest() for text in texts]
    else:
        md5s = []
        for i, text in enumerate(texts):
            text_no_dup = text + f"_{i}-{i}_{i}"
            md5s.append(hashlib.md5(text_no_dup.encode("utf-8")).hexdigest())
        return md5s


class BaseTextDataset(Dataset):
    """
    Base class for text datasets.
    """

    def __init__(
        self,
        text: list[str],
        sample_ratio: float | None = None,
        dedup: bool = True,
        remove_non_english: bool = True,
    ) -> None:
        # Clean data
        if remove_non_english:
            text = [t for t in text if is_pure_english(t)]
        if dedup:
            text = deduplicate(text, key=None)

        # Sample data
        text = sample_items(text, sample_ratio)

        # Build text & md5 pairs
        self.texts = sorted(text)
        self.md5s = texts_to_md5s(texts=self.texts, dedup=dedup)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"md5": self.md5s[idx], "text": self.texts[idx]}

    @staticmethod
    def collate_fn(batch):
        return {
            "md5": [item["md5"] for item in batch],
            "text": [item["text"] for item in batch],
        }


class BaseTextImageDataset(Dataset):
    """
    A simple dataset class for text and images.
    It returns a dict with keys: md5, text, image, text_feature, image_feature.
    """

    def __init__(
        self,
        data_file_path: (str | None) = None,  # each item contains "md5", "text", "image"
        text: list[str] | None = None,  # list of str, prompts
        image: list[str] | None = None,  # list of str, image paths
        dedup: bool = True,
        dedup_key: str | Callable | None = None,  # key for dedup, e.g. "text" or "prompt"
        sample_ratio: float | int | None = None,
        kwargs: dict = {},
    ) -> None:
        dedup_key = dedup_key or "text"
        if isinstance(data_file_path, str) and os.path.exists(data_file_path):
            if data_file_path.endswith(".jsonl"):
                raw_data = [json.loads(line) for line in open(data_file_path, "r", encoding="utf-8")]
                data = deduplicate(raw_data, key=dedup_key) if dedup else raw_data
        elif text is not None and image is not None:
            assert len(text) == len(image), "❌ The number of texts and images must be aligned"
            pairs = [{"text": t, "image": i} for t, i in zip(text, image)]
            if dedup:
                pairs = deduplicate(pairs, key="text")
            md5s = texts_to_md5s(texts=[p["text"] for p in pairs], dedup=True)
            data = [{"md5": m, "text": p["text"], "image": p["image"]} for m, p in zip(md5s, pairs)]
        else:
            raise ValueError("❌ Either data_file_path or text/image must be provided")

        self.data = sample_items(data, sample_ratio)

        self.image_open = kwargs.get("image_open", False)  # if to open images as PIL.Image
        self.tokenizer = kwargs.get("tokenizer", None)  # tokenizer for processing texts
        self.transform = kwargs.get("transform", None)  # image transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        md5 = self.data[idx]["md5"]

        text = self.data[idx]["text"]
        text_feature = self.tokenizer(text) if self.tokenizer else None

        image = self.data[idx]["image"]
        image_pil = Image.open(image) if self.image_open else None
        image_feature = self.transform(Image.open(image)) if self.transform else None

        return {
            "md5": md5,
            "text": text,
            "text_feature": text_feature,
            "image": image,
            "image_pil": image_pil,
            "image_feature": image_feature,
        }

    @staticmethod
    def collate_fn(batch):
        md5s = [item["md5"] for item in batch]
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]
        images_pil = [item["image_pil"] for item in batch]

        # Stack tensors if they exist
        text_features = None
        if batch[0].get("text_feature") is not None:
            text_features = torch.stack([item["text_feature"] for item in batch])

        image_features = None
        if batch[0].get("image_feature") is not None:
            image_features = torch.stack([item["image_feature"] for item in batch])

        return {
            "md5": md5s,
            "text": texts,
            "text_feature": text_features,
            "image": images,
            "image_pil": images_pil,
            "image_feature": image_features,
        }
