"""
SD3 model: Stable Diffusion 3 MMDiT with LoRA, plus its frozen VAE and text encoders.

References:
    - SD3:
        - Paper: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
        - Weights: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from diffusers.models import attention_processor
from diffusers.models.attention import JointTransformerBlock
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from transformers import (
    CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
)

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.models.base_model import BaseModel


def fa3_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs
):
    """SDPA by FlashAttention-3 for unmasked fp16/bf16 attention, by SDPA otherwise."""
    dtype = query.dtype
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")  # SDPA would cast its inputs to it
    fa3 = dtype in (torch.float16, torch.bfloat16) and query.shape[-1] <= 256
    if attn_mask is not None or dropout_p or kwargs or not fa3:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, **kwargs
        )
    from flash_attn_interface import flash_attn_func

    query, key, value = (x.to(dtype).transpose(1, 2) for x in (query, key, value))
    deterministic = torch.are_deterministic_algorithms_enabled()
    out = flash_attn_func(
        query, key, value, causal=is_causal, deterministic=deterministic
    )
    return out.transpose(1, 2)


class SD3Model(BaseModel):
    """
    FSDP-wrapped SD3 transformer with LoRA, and the frozen pipeline components.

    - self.model / self.ema: the transformer, trainable only through its LoRA weights.
    - self.vae, self.text_encoders, self.tokenizers, self.scheduler: frozen, unsharded.
    - EMA tracks only the trainable (LoRA) weights, and checkpoints hold only them.
    """

    def __init__(self, cfg_model, build_keys: tuple[str, ...] = ("model", "ema")) -> None:
        super().__init__(cfg_model, build_keys)
        self.save_trainable_only = True  # checkpoints hold LoRA weights only
        if cfg_model.get("attn_backend", "sdpa") == "fa3":
            try:
                import flash_attn_interface  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "❌ attn_backend=fa3 needs the flash-attn-3 package of pyproject.toml; "
                    "reinstall the environment, e.g. with uv sync"
                ) from e
            # The diffusers attention processors call F.scaled_dot_product_attention.
            functions = {**vars(F), "scaled_dot_product_attention": fa3_attention}
            attention_processor.F = SimpleNamespace(**functions)
        path = cfg_model.path

        # Frozen components; the VAE decodes in fp32 and the text encoders run in fp16.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            path, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            path, subfolder="vae", torch_dtype=DTYPE_MAP[cfg_model.vae_dtype]
        )
        self.vae.to(self.device).requires_grad_(False)
        self.tokenizers = [
            CLIPTokenizer.from_pretrained(path, subfolder="tokenizer"),
            CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2"),
            T5TokenizerFast.from_pretrained(path, subfolder="tokenizer_3"),
        ]
        self.text_encoders = [
            CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder"),
            CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2"),
            T5EncoderModel.from_pretrained(path, subfolder="text_encoder_3"),
        ]
        text_dtype = DTYPE_MAP[cfg_model.text_encoder_dtype]
        for text_encoder in self.text_encoders:
            text_encoder.to(self.device, dtype=text_dtype).requires_grad_(False).eval()
        print(f"✅ Loaded SD3 VAE, text encoders, and scheduler from <{path}>")

    def build_model(self) -> nn.Module:
        transformer = SD3Transformer2DModel.from_pretrained(
            self.cfg_model.path, subfolder="transformer", torch_dtype=torch.float32
        )
        transformer.requires_grad_(False)
        lora = self.cfg_model.lora
        lora_config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            init_lora_weights=lora.init,
            target_modules=list(lora.target_modules),
        )
        return get_peft_model(transformer, lora_config)

    def wrap_modules(self) -> tuple:
        return (JointTransformerBlock,)

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Step the trainable EMA params; the frozen base weights stay untouched."""
        if self.ema is None:
            raise ValueError("EMA model is not built")
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            if p.requires_grad:
                ema_p.add_((1 - decay) * (p.data - ema_p))

    @torch.no_grad()
    def encode_prompt(
        self, prompts: list[str], max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with the two CLIP encoders and T5, as SD3 DreamBooth does."""
        clip_embeds, pooled_embeds = [], []
        for tokenizer, text_encoder in zip(self.tokenizers[:2], self.text_encoders[:2]):
            input_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output = text_encoder(input_ids.to(self.device), output_hidden_states=True)
            clip_embeds.append(output.hidden_states[-2])
            pooled_embeds.append(output.text_embeds)
        clip_embeds = torch.cat(clip_embeds, dim=-1)
        pooled_embeds = torch.cat(pooled_embeds, dim=-1)

        input_ids = self.tokenizers[2](
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        t5_embeds = self.text_encoders[2](input_ids.to(self.device))[0]

        padding = t5_embeds.shape[-1] - clip_embeds.shape[-1]
        clip_embeds = nn.functional.pad(clip_embeds, (0, padding))
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)
        return prompt_embeds, pooled_embeds

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images in [0, 1]."""
        config = self.vae.config
        latents = latents / config.scaling_factor + config.shift_factor
        images = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return (images / 2 + 0.5).clamp(0, 1)
