# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Adapted from the local finite-difference-flow-optimization/fdfo/hpsv2_reward.py.
# Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.
"""Local HPSv2.1 with the source's float pixels and BF16 scoring autocast."""

from pathlib import Path

import torch
import torch.nn.functional as F


class TwinTiltHPSv2:
    """Score normalized feature similarity without quantizing training images."""

    metrics = ("hpsv2",)

    def __init__(self, cfg_reward):
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        self.name = "hpsv2"
        self.cfg_reward = cfg_reward
        self.device = torch.device("cuda", torch.cuda.current_device())
        checkpoint = Path(cfg_reward.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing local HPSv2.1 checkpoint: {checkpoint}")
        model, _, _ = create_model_and_transforms(
            model_name="ViT-H-14", pretrained=str(checkpoint), precision="amp",
            device=torch.device("cpu"), light_augmentation=True, output_dict=True,
        )
        size = model.visual.image_size
        self.image_size = int(size[0] if isinstance(size, tuple) else size)
        self.image_mean = self.rgb_tuple(
            getattr(model.visual, "image_mean", None),
            (0.48145466, 0.4578275, 0.40821073),
        )
        self.image_std = self.rgb_tuple(
            getattr(model.visual, "image_std", None),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.model = model.eval().requires_grad_(False).to(
            self.device, dtype=torch.float32
        )
        self.tokenizer = get_tokenizer("ViT-H-14")

    @staticmethod
    def rgb_tuple(value, fallback):
        if value is None:
            return fallback
        if isinstance(value, (int, float)):
            return (float(value),) * 3
        return tuple(float(item) for item in value)

    def preprocess(self, images):
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        height, width = images.shape[-2:]
        scale = self.image_size / float(max(height, width))
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
        if (new_height, new_width) != (height, width):
            images = F.interpolate(
                images, size=(new_height, new_width), mode="bicubic",
                align_corners=False, antialias=True,
            )
        pad_height = self.image_size - new_height
        pad_width = self.image_size - new_width
        if pad_height or pad_width:
            left, top = pad_width // 2, pad_height // 2
            images = F.pad(
                images, (left, pad_width - left, top, pad_height - top), value=0.0
            )
        mean = torch.tensor(
            self.image_mean, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.image_std, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        return (images - mean) / std

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True):
        if images.ndim != 4 or images.shape[1] != 3 or len(images) != len(prompts):
            raise ValueError("HPSv2 requires matching prompts and NCHW RGB images")
        pixels = self.preprocess(images)
        text = self.tokenizer(prompts).to(self.device, non_blocking=True)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(pixels, text)
            logits = output["image_features"] @ output["text_features"].T
            scores = logits.diagonal()
        return {"hpsv2": scores.float().tolist()}
