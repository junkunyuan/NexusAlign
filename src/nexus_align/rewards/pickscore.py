"""
PickScore reward: human preference score of an image for its prompt, with CLIP-H.

References:
    - PickScore:
        - Paper: Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image
          Generation (NeurIPS 2023)
        - Weights: https://huggingface.co/yuvalkirstain/PickScore_v1
"""

import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPModel, CLIPProcessor

from nexus_align.rewards.remote import to_uint8


def resize_matrix(size: int, new_size: int, like: torch.Tensor) -> torch.Tensor:
    """Get the (size, new_size) matrix of the antialiased bicubic resize on one axis."""
    eye = torch.eye(size, device=like.device, dtype=like.dtype).view(size, 1, 1, size)
    out = F.interpolate(eye, size=(1, new_size), mode="bicubic", antialias=True)
    return out.view(size, new_size)


class PickScore:
    """PickScore, divided by 26 into about [0, 1] for training as in Flow-GRPO."""

    metrics = ("pickscore",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.normalize = cfg_reward.normalize  # off for evaluation
        self.device = torch.device("cuda", torch.cuda.current_device())
        # The PIL image processor matches Flow-GRPO's, but the torchvision one does not.
        self.processor = CLIPProcessor.from_pretrained(
            cfg_reward.processor_path, use_fast=False
        )
        self.model = CLIPModel.from_pretrained(cfg_reward.model_path)
        self.model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = [Image.fromarray(image) for image in to_uint8(images)]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        scores = self.score(pixel_values.to(self.device), self.embed_prompts(prompts))
        return {"pickscore": scores.tolist()}

    @torch.no_grad()
    def embed_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Return the normalized text embeddings of the prompts."""
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embs = self.model.get_text_features(**text_inputs).pooler_output
        return text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

    def score(
        self, pixel_values: torch.Tensor, text_embs: torch.Tensor
    ) -> torch.Tensor:
        """Score the preprocessed images with their normalized text embeddings."""
        image_embs = self.model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs.pooler_output
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T).diag()
        if self.normalize:
            scores = scores / 26
        return scores

    def score_grad(self, images: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        """Score NCHW images in [0, 1], with gradients that flow back to the images."""
        # Resize the short side to 224 (bicubic) and center crop, as the processor does.
        # Use one resize matrix per axis, because the backward of interpolate is not
        # deterministic.
        sizes = images.shape[-2:]
        rows, cols = (resize_matrix(n, 224 * n // min(sizes), images) for n in sizes)
        images = TF.center_crop(rows.T @ images @ cols, 224)
        image_processor = self.processor.image_processor
        mean, std = image_processor.image_mean, image_processor.image_std
        return self.score(TF.normalize(images, mean, std), text_embs)
