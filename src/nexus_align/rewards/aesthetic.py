"""Aesthetic reward: LAION aesthetic predictor, an MLP on CLIP-L image embeddings.

References:
    - Improved aesthetic predictor:
        - Official code:
          https://github.com/christophschuhmann/improved-aesthetic-predictor
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/aesthetic_scorer.py)
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class MLP(nn.Module):
    """Aesthetic head; the dropout layers keep the state dict keys of the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class Aesthetic:
    """Aesthetic score on its raw scale of about 1 to 10."""

    metrics = ("aesthetic",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.device = torch.device("cuda", torch.cuda.current_device())
        # The PIL image processor matches Flow-GRPO's; the torchvision one does not.
        self.processor = CLIPProcessor.from_pretrained(
            cfg_reward.clip_path, use_fast=False
        )
        self.clip = CLIPModel.from_pretrained(cfg_reward.clip_path)
        self.clip.to(self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.mlp = MLP()
        self.mlp.load_state_dict(torch.load(cfg_reward.mlp_path, map_location="cpu"))
        self.mlp.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=torch.float32) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs).pooler_output
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return {"aesthetic": self.mlp(embed).squeeze(1).tolist()}
