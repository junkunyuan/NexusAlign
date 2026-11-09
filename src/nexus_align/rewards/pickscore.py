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
from transformers import AutoProcessor, CLIPModel, CLIPProcessor

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
        self.normalize = cfg_reward.get("normalize", True)  # off for evaluation
        self.image_input = cfg_reward.get("image_input", "pil")
        if self.image_input not in ("pil", "tensor"):
            raise ValueError("PickScore image_input must be pil or tensor")
        self.device = torch.device("cuda", torch.cuda.current_device())
        local_files_only = cfg_reward.get("local_files_only", False)
        processor = AutoProcessor if self.image_input == "tensor" else CLIPProcessor
        kwargs = {} if self.image_input == "tensor" else {"use_fast": False}
        self.processor = processor.from_pretrained(
            cfg_reward.processor_path, local_files_only=local_files_only, **kwargs
        )
        self.model = CLIPModel.from_pretrained(
            cfg_reward.model_path, local_files_only=local_files_only
        )
        self.model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        if self.image_input == "pil":
            images = [Image.fromarray(image) for image in to_uint8(images)]
            pixels = self.processor(images=images, return_tensors="pt").pixel_values
            scores = self.score(pixels.to(self.device), self.embed_prompts(prompts))
            return {"pickscore": scores.tolist()}

        # Float-input processing follows local fdfo/{rewards,pickscore}.py.
        # Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
        # See licenses/FDFO-LICENSE.txt (NVIDIA Source Code License v1).
        image_inputs = self.processor(
            images=images.to(torch.float32), padding=True, truncation=True,
            max_length=77, images_kwargs={"do_rescale": False}, return_tensors="pt",
        )
        prompts = [str(prompt).strip() for prompt in prompts]
        unique = list(dict.fromkeys(prompts))
        lookup = {prompt: index for index, prompt in enumerate(unique)}
        inverse = torch.tensor([lookup[p] for p in prompts], device=self.device)
        text_inputs = self.processor(
            text=unique, padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        )
        image_inputs = {
            k: (torch.stack(v) if isinstance(v, list) else v).to(self.device)
            for k, v in image_inputs.items()
        }
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = getattr(image_embs, "pooler_output", image_embs).float()
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = getattr(text_embs, "pooler_output", text_embs).float()
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        scores = torch.linalg.vecdot(text_embs.index_select(0, inverse), image_embs)
        scores = self.model.logit_scale.exp() * scores
        if self.normalize:
            scores = scores / 26
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
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = getattr(text_embs, "pooler_output", text_embs)
        return text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

    def score(
        self, pixel_values: torch.Tensor, text_embs: torch.Tensor
    ) -> torch.Tensor:
        """Score the preprocessed images with their normalized text embeddings."""
        image_embs = self.model.get_image_features(pixel_values=pixel_values)
        image_embs = getattr(image_embs, "pooler_output", image_embs)
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
