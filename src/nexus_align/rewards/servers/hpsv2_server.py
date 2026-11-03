"""
HPSv2.1 reward server: human preference score with CLIP-H, under gunicorn.

It runs in .venv-hpsv2, because the open_clip copy of HPSv2 needs protobuf<4.
Launch it with scripts/tools/reward_server.sh.

References:
    - HPSv2:
        - Official code: https://github.com/tgxs002/HPSv2
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/hpsv2_scorer.py)
"""

import numpy as np
import torch
import torchvision.transforms.functional as F
from flask import Flask
from torchvision.transforms import InterpolationMode
from hpsv2.src.open_clip import create_model, get_tokenizer

from common import pickle_route


def preprocess(images: torch.Tensor, image_size: int, mean, std) -> torch.Tensor:
    """Resize the longer side to image_size, pad to a square, and normalize."""
    height, width = images.shape[-2:]
    scale = image_size / float(max(height, width))
    if scale != 1.0:
        new_size = tuple(round(dim * scale) for dim in (height, width))
        images = F.resize(images, new_size, InterpolationMode.BICUBIC)
        pad_h = image_size - new_size[0]
        pad_w = image_size - new_size[1]
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        images = F.pad(images, padding=padding, fill=0)
    return F.normalize(images, mean=mean, std=std)


def create_app(checkpoint_path: str) -> Flask:
    """Build the app. A POST of pickled {"images", "prompts"} returns {"scores"}."""
    # The HPS checkpoint holds all weights, so do not load the CLIP-H weights first.
    model = create_model(
        "ViT-H-14",
        precision="amp",
        device="cuda",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        output_dict=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    tokenizer = get_tokenizer("ViT-H-14")
    image_size = model.visual.image_size
    image_size = image_size[0] if isinstance(image_size, tuple) else image_size
    mean, std = model.visual.image_mean, model.visual.image_std
    app = Flask(__name__)

    def inference(data):
        images = torch.from_numpy(np.ascontiguousarray(data["images"])).cuda()
        images = images.permute(0, 3, 1, 2).float() / 255  # NHWC -> NCHW
        with torch.no_grad():
            image = preprocess(images, image_size, mean, std)
            text = tokenizer(data["prompts"]).cuda()
            outputs = model(image, text)
            logits = outputs["image_features"] @ outputs["text_features"].T
        return {"scores": torch.diagonal(logits).tolist()}

    pickle_route(app, "/", inference)
    return app
