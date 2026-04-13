"""HPSv3 reward model for evaluating image generation."""

import os
import torch

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.core.base_reward_model import BaseRewardModel


class HPSv3(BaseRewardModel):
    """
    HPSv3 reward model for evaluating image generation.

    References:
        - HPSv3 paper: https://arxiv.org/pdf/2508.03789.
        - Official repo: https://github.com/MizzenAI/HPSv3.
        - Checkpoint: https://huggingface.co/MizzenAI/HPSv3.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("HPSv3", device, kwargs)

        # Only support bf16
        assert self.model_dtype == torch.bfloat16

        self.model = self.load_model()

        self.dataset_kwargs = {}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> torch.nn.Module:
        """
        Load model.

        Follow the repo: https://huggingface.co/MizzenAI/HPSv3.
        """

        from hpsv3 import HPSv3RewardInferencer
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLVisionBlock,
            VisionMlp,
            Qwen2VLDecoderLayer,
            Qwen2MLP,
        )

        # Load components
        model_path = os.path.join(self.model_path, "HPSv3.safetensors")
        print(f"⏳ Loading {self.model_name} model from <{model_path}>")
        model = HPSv3RewardInferencer(checkpoint_path=model_path, device=self.device)
        model.model = model.model.to(dtype=self.model_dtype)

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            model.model = fsdp_wrap(
                model=model.model,
                wrap_modules=(
                    Qwen2VLVisionBlock,
                    VisionMlp,
                    Qwen2VLDecoderLayer,
                    Qwen2MLP,
                ),
                param_dtype=self.model_dtype,
                strategy=self.fsdp_kwargs["strategy"],
                cpu_offload=self.fsdp_kwargs["cpu_offload"],
                model_name=self.model_name,
            )
        else:
            model = model.to(device=self.device, dtype=self.model_dtype)

        # Set model mode
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

        Follow the repo: https://huggingface.co/MizzenAI/HPSv3.
        
        Args:
            data (`dict`):
                image (`list`): List of paths to images.
                text (`list`): List of text strings.
            return_tensor (`bool`): If true, return a tensor; else return a list.
    
        Returns:
            `list` | `torch.Tensor`: Scores between each pair of images and texts.
        """
        images = data["image"]
        prompts = data["text"]

        result = self.model.reward(images, prompts)
        scores = result[:, 0]

        if return_tensor:
            return scores.contiguous()  
        else:
            return scores.cpu().tolist()
