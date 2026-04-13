"""HPSv2 reward model for evaluating image generation."""

import os
import torch

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.core.base_reward_model import BaseRewardModel


class HPSv2(BaseRewardModel):
    """
    HPSv2 reward model for evaluating image generation.

    References:
        - HPSv2 paper: https://arxiv.org/pdf/2306.09341.
        - Official repo: https://github.com/tgxs002/HPSv2.
        - Checkpoint: https://huggingface.co/xswu/HPSv2.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("HPSv2", device, kwargs)

        self.model, self.preprocess_val, self.tokenizer = self.load_model()

        self.dataset_kwargs = {
            "tokenizer": self.tokenizer,
            "transform": self.preprocess_val,
        }

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, callable, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://github.com/tgxs002/HPSv2/blob/master/hpsv2/evaluation.py#L259.
        """
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        # Load components
        hpsv2_path = os.path.join(self.model_path, "HPS_v2.1_compressed.pt")
        print(f"⏳ Loading {self.model_name} model from <{hpsv2_path}>")
        model, _, preprocess_val = create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained=hpsv2_path,
            precision="amp",
            light_augmentation=True,
            output_dict=True,
        )

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            from hpsv2.src.open_clip.transformer import ResidualAttentionBlock

            model = fsdp_wrap(
                model=model,
                wrap_modules=(ResidualAttentionBlock,),
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

        tokenizer = get_tokenizer("ViT-H-14")

        return model, preprocess_val, tokenizer

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of (image, text) pairs.

        Follow the repo: https://github.com/tgxs002/HPSv2/blob/master/hpsv2/evaluation.py#L132

        Another implementation yields similar results but does not support batch inference:
            import hpsv2
            scores = [hpsv2.score(i, t, hps_version="v2.1") for i, t in zip(img_paths, txt)]
        
        Args:
            data (`dict`):
                text_feature (`Tensor`): Encoded text of shape [B, 1, 77].
                image_feature (`Tensor`): Preprocessed images of shape [B, 3, 224, 224].
                or text (`list`): List of text strings.
                or image_pil (`list`): List of PIL.Image objects.
            return_tensor (`bool`): If true, return a tensor; else return a list.

        Returns:
            `list` | `torch.Tensor`: Scores between each pair of images and texts.
        """
        # Embed text
        if "text_feature" in data:
            text_features = data["text_feature"].squeeze(1).to(self.device)
        elif "text" in data:
            text_features = []
            for t in data["text"]:
                text_feature = self.tokenizer(t)
                text_features.append(text_feature)
            text_features = torch.stack(text_features)
        else:
            raise ValueError(f"❌ Invalid data: {data}")
        text_features = text_features.squeeze(1).to(self.device)

        # Embed images
        if "image_feature" in data:
            image_features = data["image_feature"].to(self.device)
        elif "image_pil" in data:
            image_features = []
            for img in data["image_pil"]:
                image_feature = self.preprocess_val(img)
                image_features.append(image_feature)
            image_features = torch.stack(image_features).to(self.device)
        else:
            raise ValueError(f"❌ Invalid data: {data}")

        # Calculate scores
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            res = self.model(image_features, text_features)
            logits = res["image_features"] @ res["text_features"].T
            scores = torch.diagonal(logits)

        if return_tensor:
            return scores.contiguous()  
        else:
            return scores.cpu().tolist()
