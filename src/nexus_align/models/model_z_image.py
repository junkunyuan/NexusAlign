"""Z-Image model for image generation."""

import os

import torch
from torch.nn.utils.rnn import pad_sequence

from diffusers import ZImageTransformer2DModel, AutoencoderKL
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
from transformers import AutoModel, AutoTokenizer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_model import BaseModel
from nexus_align.engine.distributed import all_reduce_tensor
from nexus_align.engine.fsdp import fsdp_wrap, activation_wrap


class ZImageModel(BaseModel):
    """
    Z-Image Model for image generation.

    Z-Image is a DiT-based text-to-image model that uses Qwen3 as the text
    encoder with chat-template prompt formatting, supporting multilingual
    text rendering and precise image generation.

    References:
        - Checkpoint: https://huggingface.co/Z-a-o/Z-Image-Turbo
        - Pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/z_image
    """

    def __init__(
        self,
        device: torch.device,
        model_dtype: str,
        kwargs: dict,
    ) -> None:
        self.model_name = "ZImage"
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
        self.text_encoder, self.tokenizer = self.load_text_encoder()

    def get_trainable_module(self):
        """Return the main trainable module (for BaseModel interface)."""
        return self.model

    def get_trainable_params(self):
        """Return trainable parameters (for BaseModel interface)."""
        return self.params_train

    def load_model(self) -> tuple[ZImageTransformer2DModel, tuple, list]:
        """
        Load transformer model and wrap with FSDP.

        Returns:
            (model, wrap_modules, params_train)
        """
        subfolder = "transformer"
        print(f"⏳ Loading ZImage model from <{self.pipe_path}>/{subfolder}")
        model = ZImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        # Check for pad token parameters before FSDP wrapping.
        # FSDP1 FlatParameter views corrupt these small parameters, causing
        # CUDA advanced-indexing kernel failures in _prepare_sequence.
        needs_pad_token_fix = any(
            hasattr(model, n) for n in ("x_pad_token", "cap_pad_token")
        )

        wrap_modules = (ZImageTransformerBlock,)
        model = fsdp_wrap(
            model=model,
            wrap_modules=wrap_modules,
            param_dtype=self.model_dtype,
            strategy=self.fsdp_strategy,
            cpu_offload=self.fsdp_cpu_offload,
            model_name=self.model_name,
        )

        if needs_pad_token_fix:
            self._patch_prepare_sequence(model)

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

    def load_ref_model(self) -> FSDP:
        """Load a frozen reference model with FSDP + CPU offload for KL computation."""
        subfolder = "transformer"
        print(f"⏳ Loading ZImage reference model from <{self.pipe_path}>/{subfolder}")
        ref_model = ZImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        ref_model.requires_grad_(False)
        ref_model.eval()

        needs_pad_token_fix = any(
            hasattr(ref_model, n) for n in ("x_pad_token", "cap_pad_token")
        )

        wrap_modules = (ZImageTransformerBlock,)
        ref_model = fsdp_wrap(
            model=ref_model,
            wrap_modules=wrap_modules,
            param_dtype=self.model_dtype,
            strategy=self.fsdp_strategy,
            cpu_offload=True,
            model_name="ZImage_ref",
        )

        if needs_pad_token_fix:
            self._patch_prepare_sequence(ref_model)

        print("✅ Prepared reference model: ZImage_ref (frozen, CPU offload)")
        torch.cuda.empty_cache()

        return ref_model

    @staticmethod
    def _patch_prepare_sequence(model) -> None:
        """Replace ``_prepare_sequence`` with an FSDP-safe variant.

        FSDP1 stores ``x_pad_token`` / ``cap_pad_token`` as views into a
        FlatParameter whose underlying storage is much larger than the
        logical shape.  The original advanced-indexing assignment
        ``feats_cat[mask] = pad_token`` triggers a CUDA kernel that
        mis-reads the storage size, causing an INTERNAL ASSERT failure.

        This patch replaces the assignment with ``torch.where``, which
        performs element-wise selection and avoids the problematic kernel.
        """
        model_cls = type(model._fsdp_wrapped_module)

        def _prepare_sequence(
            self,
            feats: list[torch.Tensor],
            pos_ids: list[torch.Tensor],
            inner_pad_mask: list[torch.Tensor],
            pad_token: torch.nn.Parameter,
            noise_mask: list[list[int]] | None = None,
            device: torch.device = None,
        ):
            item_seqlens = [len(f) for f in feats]
            max_seqlen = max(item_seqlens)
            bsz = len(feats)

            feats_cat = torch.cat(feats, dim=0)
            mask = torch.cat(inner_pad_mask)
            if mask.any():
                pad_val = pad_token.detach().expand_as(feats_cat)
                feats_cat = torch.where(mask.unsqueeze(1), pad_val, feats_cat)
            feats = list(feats_cat.split(item_seqlens, dim=0))

            freqs_cis = list(
                self.rope_embedder(torch.cat(pos_ids, dim=0)).split(
                    [len(p) for p in pos_ids], dim=0,
                )
            )

            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
            freqs_cis = pad_sequence(
                freqs_cis, batch_first=True, padding_value=0.0,
            )[:, : feats.shape[1]]

            attn_mask = torch.zeros(
                (bsz, max_seqlen), dtype=torch.bool, device=device,
            )
            for i, seq_len in enumerate(item_seqlens):
                attn_mask[i, :seq_len] = 1

            noise_mask_tensor = None
            if noise_mask is not None:
                noise_mask_tensor = pad_sequence(
                    [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                    batch_first=True, padding_value=0,
                )[:, : feats.shape[1]]

            return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

        model_cls._prepare_sequence = _prepare_sequence

    def load_vae(self) -> AutoencoderKL:
        """Load VAE module."""
        subfolder = "vae"
        print(f"⏳ Loading ZImage VAE from <{self.pipe_path}>/{subfolder}")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        ).to(self.device)

        vae.eval()
        vae.requires_grad_(False)

        print("✅ Prepared VAE: ZImage_VAE (eval mode)")

        return vae

    def load_text_encoder(self) -> tuple:
        """Load Qwen3 text encoder and tokenizer."""
        subfolder = "text_encoder"
        print(f"⏳ Loading ZImage {subfolder} from <{self.pipe_path}>/{subfolder}")
        text_encoder = AutoModel.from_pretrained(
            self.pipe_path, subfolder=subfolder,
        )

        subfolder = "tokenizer"
        print(f"⏳ Loading ZImage {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer = AutoTokenizer.from_pretrained(self.pipe_path, subfolder=subfolder)

        if self.text_encoder_offload:
            text_encoder.to("cpu")
        else:
            text_encoder.to(self.device)

        text_encoder.eval()
        text_encoder.requires_grad_(False)

        print("✅ Prepared text encoder: Qwen3 (eval mode)")

        return text_encoder, tokenizer

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str | list[str],
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> dict:
        """
        Encode prompt using Qwen3 text encoder.

        Applies a chat template with thinking enabled and extracts the
        second-to-last hidden states. Returns variable-length embeddings
        as a list (one tensor per prompt, no padding).
        """
        dtype = dtype or self.model_dtype
        device = device or self.device

        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder_offload:
            self.text_encoder.to(device)

        # Apply chat template to each prompt
        formatted = []
        for p in prompt:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted.append(text)

        # Tokenize and encode
        tokens = self.tokenizer(
            formatted,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoder_output = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_output.hidden_states[-2]

        # Extract only valid (non-padding) tokens per prompt
        prompt_masks = tokens.attention_mask.bool()
        prompt_embeds = []
        for i in range(len(hidden_states)):
            valid = hidden_states[i][prompt_masks[i]].to(dtype=dtype)
            prompt_embeds.append(valid)

        if self.text_encoder_offload:
            self.text_encoder.to("cpu")

        text_embed = {
            "prompt_embeds": prompt_embeds,
        }

        torch.cuda.empty_cache()

        return text_embed
