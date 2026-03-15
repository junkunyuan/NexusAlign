"""Stable Diffusion 3 model for image generation."""

import os

import torch
import torch.nn.functional as F

from diffusers import SD3Transformer2DModel, AutoencoderKL, StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3SingleTransformerBlock
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_model import BaseModel
from nexus_align.engine.distributed import all_reduce_tensor
from nexus_align.engine.fsdp import fsdp_wrap, activation_wrap


class SD3Model(BaseModel):
    """
    Stable Diffusion 3 Model for image generation.

    SD3 is an MMDiT-based text-to-image model using triple text encoders
    (CLIP-L, CLIP-G, T5-XXL) and flow-matching diffusion.

    References:
        - Paper: https://arxiv.org/abs/2403.03206
        - Checkpoint: https://huggingface.co/stabilityai/stable-diffusion-3-medium
    """

    def __init__(
        self,
        device: torch.device,
        model_dtype: str,
        kwargs: dict,
    ) -> None:
        self.model_name = "SD3"
        self.data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        self.model_path = kwargs["model"]["path"]
        self.pipe_path = os.path.join(self.data_and_model_dir, self.model_path)
        self.safetensors_file = kwargs["model"].get(
            "safetensors_file", "sd3_medium_incl_clips_t5xxlfp16.safetensors"
        )
        self.model_dtype = DTYPE_MAP[model_dtype]
        self.device = device
        self.mode = kwargs["model"]["mode"]

        self.fsdp_strategy = kwargs["model"]["fsdp"]["fsdp_strategy"]
        self.fsdp_cpu_offload = kwargs["model"]["fsdp"]["fsdp_cpu_offload"]
        self.activation_ckpt = kwargs["model"]["fsdp"]["activation_ckpt"]
        self.text_encoder_offload = kwargs["model"]["fsdp"]["text_encoder_offload"]

        # Load all components from single safetensors file
        self._load_from_single_file()

    def _load_from_single_file(self) -> None:
        """Load all components from SD3 single safetensors file."""
        safetensors_path = os.path.join(self.pipe_path, self.safetensors_file)
        print(f"⏳ Loading SD3 pipeline from <{safetensors_path}>")
        pipe = StableDiffusion3Pipeline.from_single_file(
            safetensors_path, torch_dtype=self.model_dtype,
        )
        print(f"✅ Loaded SD3 pipeline from single file")

        # Extract and prepare each component
        self.model, self.wrap_modules, self.params_train = self._prepare_transformer(
            pipe.transformer
        )
        self.vae = self._prepare_vae(pipe.vae)
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self._prepare_text_encoders()

        del pipe
        torch.cuda.empty_cache()

    def get_trainable_module(self):
        """Return the main trainable module (for BaseModel interface)."""
        return self.model

    def get_trainable_params(self):
        """Return trainable parameters (for BaseModel interface)."""
        return self.params_train

    def _prepare_transformer(
        self, transformer: SD3Transformer2DModel
    ) -> tuple[SD3Transformer2DModel, tuple, list]:
        """Wrap transformer with FSDP and set mode."""
        print(f"⏳ Preparing SD3 transformer with FSDP")
        wrap_modules = (SD3SingleTransformerBlock,)
        model = fsdp_wrap(
            model=transformer,
            wrap_modules=wrap_modules,
            param_dtype=self.model_dtype,
            strategy=self.fsdp_strategy,
            cpu_offload=self.fsdp_cpu_offload,
            model_name=self.model_name,
        )

        if self.activation_ckpt:
            activation_wrap(model, wrap_modules, model_name=self.model_name)

        if self.mode == "train":
            model.train()
        elif self.mode == "eval":
            model.eval()
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")

        para_train = [p for p in model.parameters() if p.requires_grad]
        train_n = sum(p.numel() for p in para_train)
        total_n = sum(p.numel() for p in model.parameters())
        train_n = all_reduce_tensor(train_n, op="sum") / 1e9
        total_n = all_reduce_tensor(total_n, op="sum") / 1e9

        para_info = f"trainable params: {train_n:.4f} B / {total_n:.4f} B"
        print(f"✅ Prepared model: {self.model_name} ({self.mode} mode) ({para_info})")

        torch.cuda.empty_cache()

        return model, wrap_modules, para_train

    def _prepare_vae(self, vae: AutoencoderKL) -> AutoencoderKL:
        """Move VAE to device and freeze."""
        print(f"⏳ Preparing SD3 VAE")
        vae = vae.to(dtype=self.model_dtype, device=self.device)
        vae.eval()
        vae.requires_grad_(False)
        print("✅ Prepared VAE: SD3_VAE (eval mode)")
        return vae

    def _prepare_text_encoders(self) -> None:
        """Move text encoders to device/CPU and freeze."""
        print(f"⏳ Preparing SD3 text encoders (CLIP-L, CLIP-G, T5-XXL)")
        if self.text_encoder_offload:
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
            self.text_encoder_3.to("cpu")
        else:
            self.text_encoder.to(self.device)
            self.text_encoder_2.to(self.device)
            self.text_encoder_3.to(self.device)

        for enc in [self.text_encoder, self.text_encoder_2, self.text_encoder_3]:
            enc.eval()
            enc.requires_grad_(False)

        print("✅ Prepared text encoders: CLIP-L & CLIP-G & T5-XXL (eval mode)")

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] = None,
        prompt_3: str | list[str] = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> dict:
        """
        Encode prompt using SD3 triple encoders (CLIP-L, CLIP-G, T5-XXL).

        Produces concatenated prompt embeddings and pooled projections in the
        format expected by the SD3 transformer.
        """
        dtype = dtype or self.model_dtype
        device = device or self.device

        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt
        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]
        if isinstance(prompt_3, str):
            prompt_3 = [prompt_3]

        if self.text_encoder_offload:
            self.text_encoder.to(device)
            self.text_encoder_2.to(device)
            self.text_encoder_3.to(device)

        # CLIP-L encoding
        clip_l_input = self.tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        clip_l_output = self.text_encoder(clip_l_input, output_hidden_states=True)
        clip_l_embeds = clip_l_output.hidden_states[-2]
        clip_l_pooled = clip_l_output[0]

        # CLIP-G encoding
        clip_g_input = self.tokenizer_2(
            prompt_2, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        clip_g_output = self.text_encoder_2(clip_g_input, output_hidden_states=True)
        clip_g_embeds = clip_g_output.hidden_states[-2]
        clip_g_pooled = clip_g_output[0]

        # T5-XXL encoding
        t5_input = self.tokenizer_3(
            prompt_3, padding="max_length", max_length=256,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        t5_embeds = self.text_encoder_3(t5_input)[0]
        t5_embeds = t5_embeds.to(dtype=dtype)

        # Combine CLIP embeddings and pad to T5 dimension
        clip_embeds = torch.cat([clip_l_embeds, clip_g_embeds], dim=-1)
        clip_embeds = F.pad(
            clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1])
        )

        # Concatenate CLIP + T5 along sequence dim
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Pooled projections from both CLIP encoders
        pooled_prompt_embeds = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype, device=device)

        if self.text_encoder_offload:
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
            self.text_encoder_3.to("cpu")

        text_embed = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        torch.cuda.empty_cache()

        return text_embed
