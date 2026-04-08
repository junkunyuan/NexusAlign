"""Qwen-Image model for image generation."""

import os

import torch

from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_model import BaseModel
from nexus_align.engine.distributed import all_reduce_tensor
from nexus_align.engine.fsdp import fsdp_wrap, activation_wrap

# Qwen-Image uses a system prompt template for text encoding.
# The template instructs the model to describe image attributes, following
# the official QwenImagePipeline convention.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
# Number of leading tokens from the system prompt prefix to drop
TEMPLATE_DROP_IDX = 34


class QwenImageModel(BaseModel):
    """
    Qwen-Image Model for image generation.

    Qwen-Image is a DiT-based text-to-image model that uses Qwen2.5-VL as
    the text encoder, achieving strong multilingual text rendering and
    precise image generation.

    References:
        - Checkpoint: https://huggingface.co/Qwen/Qwen-Image.
        - Pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/qwenimage.
    """

    def __init__(
        self,
        device: torch.device,
        model_dtype: str,
        kwargs: dict,
    ) -> None:
        self.model_name = "QwenImage"
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

    def load_model(self) -> tuple[QwenImageTransformer2DModel, list, list]:
        """
        Load transformer model and wrap with FSDP.

        Returns:
            (model, wrap_modules, params_train)
        """
        subfolder = "transformer"
        print(f"⏳ Loading QwenImage model from <{self.pipe_path}>/{subfolder}")
        model = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        wrap_modules = (QwenImageTransformerBlock,)
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
        print(f"⏳ Loading QwenImage reference model from <{self.pipe_path}>/{subfolder}")
        ref_model = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        )

        ref_model.requires_grad_(False)
        ref_model.eval()

        wrap_modules = (QwenImageTransformerBlock,)
        ref_model = fsdp_wrap(
            model=ref_model,
            wrap_modules=wrap_modules,
            param_dtype=self.model_dtype,
            strategy=self.fsdp_strategy,
            cpu_offload=True,
            model_name="QwenImage_ref",
        )

        print("✅ Prepared reference model: QwenImage_ref (frozen, CPU offload)")
        torch.cuda.empty_cache()

        return ref_model

    def load_vae(self) -> AutoencoderKLQwenImage:
        """Load VAE module."""
        subfolder = "vae"
        print(f"⏳ Loading QwenImage VAE from <{self.pipe_path}>/{subfolder}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            pretrained_model_name_or_path=self.pipe_path,
            subfolder=subfolder,
            torch_dtype=self.model_dtype,
        ).to(self.device)

        vae.eval()
        vae.requires_grad_(False)

        print("✅ Prepared VAE: QwenImage_VAE (eval mode)")

        return vae

    def load_text_encoder(
        self,
    ) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoTokenizer]:
        """Load Qwen2.5-VL text encoder and tokenizer."""
        subfolder = "text_encoder"
        print(f"⏳ Loading QwenImage {subfolder} from <{self.pipe_path}>/{subfolder}")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.pipe_path, subfolder=subfolder,
        )

        subfolder = "tokenizer"
        print(f"⏳ Loading QwenImage {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer = AutoTokenizer.from_pretrained(self.pipe_path, subfolder=subfolder)

        if self.text_encoder_offload:
            text_encoder.to("cpu")
        else:
            text_encoder.to(self.device)

        text_encoder.eval()
        text_encoder.requires_grad_(False)

        print("✅ Prepared text encoder: Qwen2_5_VL (eval mode)")

        return text_encoder, tokenizer

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str | list[str],
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> dict:
        """
        Encode prompt using Qwen2.5-VL text encoder.

        Applies a system prompt template and extracts the last hidden state,
        dropping the system-prompt prefix tokens to get clean prompt embeddings.
        Returns variable-length embeddings with an attention mask for padding.
        """
        dtype = dtype or self.model_dtype
        device = device or self.device

        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder_offload:
            self.text_encoder.to(device)

        # Wrap each prompt in the Qwen-Image system template
        txt = [PROMPT_TEMPLATE.format(p) for p in prompt]
        max_length = 1024 + TEMPLATE_DROP_IDX
        txt_tokens = self.tokenizer(
            txt,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Extract last-layer hidden states from Qwen2.5-VL
        encoder_output = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_output.hidden_states[-1]

        # Extract non-padding hidden states, then drop the system-prompt prefix
        bool_mask = txt_tokens.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)
        split_hidden = [h[TEMPLATE_DROP_IDX:] for h in split_hidden]

        # Pad to uniform length and build attention mask
        attn_masks = [
            torch.ones(h.size(0), dtype=torch.long, device=device)
            for h in split_hidden
        ]
        max_seq_len = max(h.size(0) for h in split_hidden)
        prompt_embeds = torch.stack([
            torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))])
            for h in split_hidden
        ])
        prompt_embeds_mask = torch.stack([
            torch.cat([m, m.new_zeros(max_seq_len - m.size(0))])
            for m in attn_masks
        ])

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # None mask means all tokens are valid (no padding needed)
        if prompt_embeds_mask.all():
            prompt_embeds_mask = None

        if self.text_encoder_offload:
            self.text_encoder.to("cpu")

        text_embed = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }

        torch.cuda.empty_cache()

        return text_embed
