"""FLUX model for image generation."""

import os

import torch

from diffusers import AutoencoderKL
from diffusers import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.models.transformers.transformer_flux import (
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)

from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_model import BaseModel
from nexus_align.engine.distributed import all_reduce_tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from nexus_align.engine.fsdp import fsdp_wrap, activation_wrap


class FluxModel(BaseModel):
    """
    FLUX Model for image generation.

    FLUX is a flow-based transformer for text-to-image generation.

    References:
        - Official repo: https://github.com/black-forest-labs/flux.
        - Checkpoint: https://huggingface.co/black-forest-labs/FLUX.1-dev.
    """

    def __init__(
        self,
        device: torch.device,
        model_dtype: str,
        kwargs: dict,
    ) -> None:
        self.model_name = "FLUX"
        self.data_and_model_dir = kwargs["common"]["data_and_model_dir"]
        self.model_path = kwargs["model"]["path"]
        self.pipe_path = os.path.join(self.data_and_model_dir, self.model_path)
        self.model_dtype = DTYPE_MAP[model_dtype]
        self.device = device
        self.mode = kwargs["model"]["mode"]

        self.fsdp_strategy = kwargs["model"]["fsdp"]["fsdp_strategy"]
        self.fsdp_cpu_offload = kwargs["model"]["fsdp"]["fsdp_cpu_offload"]
        self.activation_ckpt = kwargs["model"]["fsdp"]["activation_ckpt"]
        self.text_encoder_offload = kwargs["model"]["fsdp"]["text_encoder_offload"]

        self.model, self.wrap_modules, self.params_train = self.load_model()
        self.ref_model = self.load_ref_model() if self.mode == "train" else None
        self.vae = self.load_vae()
        self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2 = (
            self.load_text_encoder()
        )

    def get_trainable_module(self):
        """Return the main trainable module (for BaseModel interface)."""
        return self.model

    def get_trainable_params(self):
        """Return trainable parameters (for BaseModel interface)."""
        return self.params_train
        
    def load_model(self) -> tuple[FluxTransformer2DModel, list, list]:
        """
        Load model and wrap with FSDP.

        Returns:
            (model, wrap_modules, params_train)
        """
        subfolder = "transformer"
        print(f"⏳ Loading FLUX model from <{self.pipe_path}>/{subfolder}")
        model = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        wrap_modules = (FluxTransformerBlock, FluxSingleTransformerBlock)
        model = fsdp_wrap(
            model=model,
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

        # Count trainable and total parameters
        para_train = [p for p in model.parameters() if p.requires_grad]
        train_n = sum(p.numel() for p in para_train)
        total_n = sum(p.numel() for p in model.parameters())
        train_n = all_reduce_tensor(train_n, op="sum") / 1e9
        total_n = all_reduce_tensor(total_n, op="sum") / 1e9

        para_info = f"trainable params: {train_n:.4f} B / {total_n:.4f} B"
        print(f"✅ Prepared model: {self.model_name} ({self.mode} mode) ({para_info})")

        torch.cuda.empty_cache()

        return model, wrap_modules, para_train

    def load_ref_model(self) -> FSDP:
        """Load a frozen reference model with FSDP + CPU offload for KL computation."""
        subfolder = "transformer"
        print(f"⏳ Loading FLUX reference model from <{self.pipe_path}>/{subfolder}")
        ref_model = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        ref_model.requires_grad_(False)
        ref_model.eval()

        wrap_modules = (FluxTransformerBlock, FluxSingleTransformerBlock)
        ref_model = fsdp_wrap(
            model=ref_model,
            wrap_modules=wrap_modules,
            param_dtype=self.model_dtype,
            strategy=self.fsdp_strategy,
            cpu_offload=True,
            model_name="FLUX_ref",
        )

        print("✅ Prepared reference model: FLUX_ref (frozen, CPU offload)")
        torch.cuda.empty_cache()

        return ref_model

    def load_vae(self) -> AutoencoderKL:
        """Load VAE module."""
        subfolder = "vae"
        print(f"⏳ Loading FLUX VAE from <{self.pipe_path}>/{subfolder}")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        ).to(self.device)

        vae.eval()
        vae.requires_grad_(False)

        print("✅ Prepared VAE: FLUX_VAE (eval mode)")

        return vae

    def load_text_encoder(
        self,
    ) -> tuple[CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast]:
        """Load text encoder module."""
        subfolder = "text_encoder"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        text_encoder = CLIPTextModel.from_pretrained(self.pipe_path, subfolder=subfolder)
        
        subfolder = "tokenizer"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer = CLIPTokenizer.from_pretrained(self.pipe_path, subfolder=subfolder)

        subfolder = "text_encoder_2"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        text_encoder_2 = T5EncoderModel.from_pretrained(self.pipe_path, subfolder=subfolder)
        
        subfolder = "tokenizer_2"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer_2 = T5TokenizerFast.from_pretrained(self.pipe_path, subfolder=subfolder)

        # Whether to offload text encoder to CPU for saving memory
        if self.text_encoder_offload:
            text_encoder.to("cpu")
            text_encoder_2.to("cpu")
        else:
            text_encoder.to(self.device)
            text_encoder_2.to(self.device)

        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        print("✅ Prepared text encoder: CLIPTextModel & T5EncoderModel (eval mode)")

        return text_encoder, text_encoder_2, tokenizer, tokenizer_2

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: str = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> dict:
        """Encode prompt."""
        dtype = dtype or self.text_encoder.dtype
        device = device or self.device
        batch_size = len(prompt)
        prompt_2 = prompt_2 or prompt

        if self.text_encoder_offload:
            self.text_encoder.to(device)
            self.text_encoder_2.to(device)

        # Encode prompt by CLIP
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids.to(device)
        pooled_prompt_embeds = self.text_encoder(
            text_input_ids, output_hidden_states=False
        )
        pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.detach().to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size, -1)

        # Encode prompt by T5
        text_input_ids = self.tokenizer_2(
            prompt_2,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=False)[
            0
        ]
        prompt_embeds = prompt_embeds.detach().to(dtype=dtype)
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        # Get text_ids
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        text_ids = text_ids.to(device=device, dtype=dtype)

        if self.text_encoder_offload:
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")

        text_embed = {
            "prompt_embed_t5": prompt_embeds,
            "prompt_embed_clip": pooled_prompt_embeds,
            "text_id": text_ids,
        }

        torch.cuda.empty_cache()

        return text_embed
