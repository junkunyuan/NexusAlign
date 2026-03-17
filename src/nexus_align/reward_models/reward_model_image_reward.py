"""ImageReward reward model for evaluating image generation."""

import os
import torch

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.core.base_reward_model import BaseRewardModel


class ImageReward(BaseRewardModel):
    """
    ImageReward reward model for evaluating image generation.

    References:
        - ImageReward paper: https://arxiv.org/pdf/2304.05977.
        - Official repo: https://huggingface.co/zai-org/ImageReward.
        - Checkpoint: https://huggingface.co/zai-org/ImageReward.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("ImageReward", device, kwargs)

        self.model = self.load_model()

        self.dataset_kwargs = {}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> torch.nn.Module:
        """
        Load model.

        Follow the repo: https://huggingface.co/zai-org/ImageReward.
        """
        import ImageReward as reward

        model_path = os.path.join(self.model_path, "ImageReward.pt")
        print(f"⏳ Loading {self.model_name} model from <{model_path}>")
        model = reward.load(model_path)

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            model = fsdp_wrap(
                model=model,
                wrap_modules=None,
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

        return model

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of (image, text) pairs.

        Follow the repo: https://huggingface.co/zai-org/ImageReward.
        
        Args:
            data (`dict`):
                image (`list`): List of paths to images.
                text (`list`): List of text strings.
            return_tensor (`bool`): Whether to return a tensor.
        """
        images = data["image"]
        prompts = data["text"]

        # NOTE: it does not support batch inference.
        scores = []
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            for img, txt in zip(images, prompts):
                score = self.model.score(txt, img)
                scores.append(score)

        if return_tensor:
            return torch.tensor(scores, device=self.device).contiguous()  
        else:
            return scores
