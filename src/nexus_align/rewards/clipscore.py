"""ClipScore reward: CLIP-L cosine similarity of an image and its prompt.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/clip_scorer.py)
"""

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
    """CLIP logit divided by its scale of 100, i.e. the cosine, as DiffusionNFT."""

    metrics = ("clipscore",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.processor = CLIPProcessor.from_pretrained(
            cfg_reward.clip_path, use_fast=False
        )
        self.model = CLIPModel.from_pretrained(cfg_reward.clip_path)
        self.model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)
        # DiffusionNFT transforms the float pixels with torchvision, not the processor.
        image_processor = self.processor.image_processor
        config = image_processor.to_dict()
        mean, std = image_processor.image_mean, image_processor.image_std
        self.transform = T.Compose([
            T.Resize(get_size(config["size"]))
            if config.get("do_resize") else nn.Identity(),
            T.CenterCrop(get_size(config["crop_size"]))
            if config.get("do_center_crop") else nn.Identity(),
            T.Normalize(mean=mean, std=std)
            if config.get("do_normalize") else nn.Identity(),
        ])

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        texts = self.processor(
            text=prompts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        pixels = self.transform(images.to(self.device, dtype=torch.float32))
        outputs = self.model(pixel_values=pixels, **texts)
        return {"clipscore": (outputs.logits_per_image.diagonal() / 100).tolist()}
