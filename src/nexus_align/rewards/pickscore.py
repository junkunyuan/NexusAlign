"""PickScore reward: CLIP-H human preference score of an image for its prompt.

References:
    - PickScore:
        - Paper: Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image
          Generation
        - Weights: https://huggingface.co/yuvalkirstain/PickScore_v1
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class PickScore:
    """PickScore, divided by 26 into about [0, 1] for training as in Flow-GRPO."""

    metrics = ("pickscore",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.normalize = cfg_reward.get("normalize", True)  # off for evaluation
        self.device = torch.device("cuda", torch.cuda.current_device())
        # The PIL image processor matches Flow-GRPO's; the torchvision one does not.
        self.processor = CLIPProcessor.from_pretrained(
            cfg_reward.processor_path, use_fast=False
        )
        self.model = CLIPModel.from_pretrained(cfg_reward.model_path)
        self.model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = [Image.fromarray(image) for image in images.transpose(0, 2, 3, 1)]
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs).pooler_output
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = self.model.get_text_features(**text_inputs).pooler_output
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T).diag()
        if self.normalize:
            scores = scores / 26
        return {"pickscore": scores.tolist()}
