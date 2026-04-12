"""AestheticPredictorV2 reward model for evaluating image generation."""

import torch

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.core.base_reward_model import BaseRewardModel


class AestheticPredictorV2(BaseRewardModel):
    """
    AestheticPredictorV2 model for evaluating image generation.
 
    References:
        - Official repo: https://github.com/christophschuhmann/improved-aesthetic-predictor.
        - Used repo: https://github.com/shunk031/simple-aesthetics-predictor.
        - Checkpoint: https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE.
    """

    def __init__(self, device: torch.device, kwargs: dict) -> None:
        super().__init__("AestheticPredictorV2", device, kwargs)

        self.model, self.processor = self.load_model()

        self.dataset_kwargs = {"image_open": True}

        print(f"✅ Prepared reward model: {self.model_name} ({self.mode} mode)")

    def load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load model.

        Follow the repo: https://github.com/shunk031/simple-aesthetics-predictor.
        """
        from transformers import CLIPProcessor
        from aesthetics_predictor import AestheticsPredictorV2Linear

        # Load components
        print(f"⏳ Loading {self.model_name} processor from <{self.model_path}>")
        processor = CLIPProcessor.from_pretrained(self.model_path)
        print(f"⏳ Loading {self.model_name} model from <{self.model_path}>")
        model = AestheticsPredictorV2Linear.from_pretrained(self.model_path)

        # Wrap model with FSDP
        if self.fsdp_kwargs["strategy"] is not None:
            from transformers.models.clip.modeling_clip import CLIPVisionTransformer

            model = fsdp_wrap(
                model=model,
                wrap_modules=(CLIPVisionTransformer,),
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

        return model, processor

    @torch.no_grad()
    def evaluate(self, data: dict, return_tensor: bool = False) -> list | torch.Tensor:
        """
        Evaluate a batch of images.

        Follow the repo: https://github.com/shunk031/simple-aesthetics-predictor.
        
        Args:
            data (`dict`):
                image_pil (`list`): List of PIL.Image objects.
            return_tensor (`bool`): If true, return a tensor; else return a list.

        Returns:
            `list` | `torch.Tensor`: Scores of the images.
        """

        images = data["image_pil"]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            scores = self.model(**inputs).logits.squeeze(1)

        if return_tensor:
            return scores.contiguous()  
        else:
            return scores.cpu().tolist()
