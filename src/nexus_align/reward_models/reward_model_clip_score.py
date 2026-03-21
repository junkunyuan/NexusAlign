"""CLIPScore reward model for evaluating image generation."""

import torch

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.core.base_reward_model import BaseRewardModel


class CLIPScore(BaseRewardModel):
    """
    CLIPScore reward model for evaluating image generation.

    References:
        - CLIPScore paper: https://arxiv.org/pdf/2104.08718.
        - Checkpoint: https://huggingface.co/openai/clip-vit-large-patch14.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("CLIPScore", device, kwargs)

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://huggingface.co/openai/clip-vit-large-patch14.
        """
        from transformers import CLIPModel, CLIPProcessor

        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")
        model = CLIPModel.from_pretrained(self.model_path)

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPMLP

            model = fsdp_wrap(
                model=model,
                wrap_modules=(CLIPEncoder, CLIPMLP),
                param_dtype=self.model_dtype,
                strategy=self.fsdp_kwargs["strategy"],
                cpu_offload=self.fsdp_kwargs["cpu_offload"],
                model_name=self.model_name,
            )
        else:
            model = model.to(device=self.device, dtype=self.model_dtype)

        if self.mode == "train":
            model.train()
        elif self.mode == "eval":
            model.eval()
            model.requires_grad_(False)
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")

        processor = CLIPProcessor.from_pretrained(self.model_path)

        return model, processor

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of (image, text) pairs.

        Follow the repo: https://huggingface.co/openai/clip-vit-large-patch14.

        Args:
            data (`dict`):
                image_pil (`list`): List of PIL.Image objects.
                text (`list`): List of text strings.
            return_tensor (`bool`): Whether to return a tensor.

        Another implementation yields similar results but does not support batch inference:
            from torchmetrics.multimodal import CLIPScore
            model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
            model = model.to(device)
            img_tensor = torch.stack([ToTensor(img) * 255. for img in image_pils])
            scores = [model(i, t) for i, t in zip(img_tensor, text)]

        Returns:
            `list` | `torch.Tensor`: Scores between each pair of images and texts.
        """
        images = data["image_pil"]
        prompts = data["text"]

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            outputs = self.model(**inputs)
        scores = outputs["logits_per_image"].diagonal()

        if return_tensor:
            return scores.contiguous()  
        else:
            return scores.cpu().tolist()
