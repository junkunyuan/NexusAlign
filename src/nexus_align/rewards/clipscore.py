"""
ClipScore reward: CLIP-L cosine similarity of an image and its prompt.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/clip_scorer.py)
"""

import math

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor


def get_size(size: dict | int) -> tuple[int, int] | int:
    """Read an image processor size as a torchvision size."""
    if isinstance(size, int):
        return (size, size)
    if "height" in size and "width" in size:
        return (size["height"], size["width"])
    if "shortest_edge" in size:
        return size["shortest_edge"]
    raise ValueError(f"❌ Invalid size: {size}")


class ClipScore:
    """CLIP logit with divisor 100 by default; divisor 1 exposes raw CoFiTi scores."""
