"""PickScore reward model for evaluating image generation."""

import os
import torch

from nexus_align.core.base_reward_model import BaseRewardModel


class PickScore(BaseRewardModel):
    """
    PickScore reward model for evaluating image generation.

    References:
        - PickScore paper: https://arxiv.org/pdf/2305.01569.
        - Official repo: https://huggingface.co/yuvalkirstain/PickScore_v1.
        - Checkpoint: https://huggingface.co/yuvalkirstain/PickScore_v1.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("PickScore", device, kwargs)

        self.processor_path = os.path.join(
            kwargs["common"]["data_and_model_dir"],
            kwargs["reward_model"]["processor_path"],
        )

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://huggingface.co/yuvalkirstain/PickScore_v1
        """
        from transformers import AutoProcessor, AutoModel

        # Load components
        print(f"⏳ Loading {self.model_name} processor from <{self.processor_path}>")
        processor = AutoProcessor.from_pretrained(self.processor_path)
        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")
        model = AutoModel.from_pretrained(self.model_path)

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            # TODO: some errors when wrapping PickScore, will be fixed later
            print("❌ Wrap PickScore with FSDP is not supported! Disable FSDP...")
        model = model.to(device=self.device, dtype=self.model_dtype)

        # Set model mode
        if self.mode == "train":
            model.train()
        elif self.mode == "eval":
            model.eval()
            model.requires_grad_(False)
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")

        return model, processor

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of (image, text) pairs.

        Args:
            data (`dict`):
                image_pil (`list`): List of PIL.Image objects.
                text (`list`): List of text strings.
            return_tensor (`bool`): If true, return a tensor; else return a list.

        Returns:
            `list` | `torch.Tensor`: Scores between each pair of images and texts.
        """
        images = data["image_pil"]
        prompts = data["text"]

        # Preprocess images and prompts
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Calculate scores
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        logits = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = torch.diagonal(logits)

        if return_tensor:
            return scores.contiguous()  
        else:
            return scores.cpu().tolist()
