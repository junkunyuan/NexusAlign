"""FLUX model for image generation."""

import json
import os

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL
from diffusers import FluxTransformer2DModel
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
from diffusers.models.transformers.transformer_flux import (
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)

from nexus_align.core.config import DTYPE_MAP
from nexus_align.core.base_model import BaseModel
from nexus_align.engine.distributed import all_reduce_tensor
from nexus_align.engine.fsdp import fsdp_wrap, activation_wrap, convert_scalar_parameters


_CONFIG_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_CONFIG_DTYPE_ALIASES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _dtype_from_config(model_path: str, subfolder: str = "transformer") -> torch.dtype | None:
    """Read torch_dtype from model_path/subfolder/config.json. Default subfolder=transformer (DiT)."""
    path = os.path.join(model_path, subfolder, "config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f).get("torch_dtype")
        if not isinstance(raw, str):
            return None
        s = raw.lower().replace("torch.", "").strip()
        s = _CONFIG_DTYPE_ALIASES.get(s, s)
        return _CONFIG_DTYPE_MAP.get(s)
    except Exception:
        return None


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
        *,
        env=None,
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
        self.use_sharded_weights = kwargs["model"]["fsdp"].get("use_sharded_weights", False)
        self.sharded_weights_dir = kwargs["model"]["fsdp"].get("sharded_weights_dir", "")
        self.use_sharded_text_encoder = kwargs["model"]["fsdp"].get(
            "use_sharded_text_encoder", False
        )
        self.t5_fsdp_shards_dir = kwargs["model"]["fsdp"].get("t5_fsdp_shards_dir", "")
        self.clip_fsdp_shards_dir = kwargs["model"]["fsdp"].get("clip_fsdp_shards_dir", "")
        self.rank = env.rank if env else None
        self.world_size = env.world_size if env else None
        if (self.use_sharded_weights or self.use_sharded_text_encoder) and env is None:
            raise ValueError("❌ env required for FSDP (use_sharded_weights or use_sharded_text_encoder)")

        self.model, self.wrap_modules, self.params_train = self.load_model()
        self.vae = self.load_vae()
        self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2 = (
            self.load_text_encoder()
        )
        cfg_model_ref = kwargs["model"].get("ref", {})
        self.ref_model = None
        if cfg_model_ref.get("enable", False):
            self.ref_model = self._load_ref_transformer(cfg_ref=cfg_model_ref)

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
        wrap_modules = (FluxTransformerBlock, FluxSingleTransformerBlock)

        if self.use_sharded_weights:
            model = self._load_sharded(
                wrap_modules=wrap_modules,
                pipe_path=self.pipe_path,
                model_dtype=self.model_dtype,
                fsdp_strategy=self.fsdp_strategy,
                fsdp_cpu_offload=self.fsdp_cpu_offload,
                sharded_weights_dir=self.sharded_weights_dir,
                model_name=self.model_name,
            )
        else:
            model = self._load_pretrained(
                wrap_modules=wrap_modules,
                pipe_path=self.pipe_path,
                model_dtype=self.model_dtype,
                fsdp_strategy=self.fsdp_strategy,
                fsdp_cpu_offload=self.fsdp_cpu_offload,
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

    def _load_pretrained(
        self,
        wrap_modules: tuple,
        *,
        pipe_path: str,
        model_dtype: torch.dtype,
        fsdp_strategy: str,
        fsdp_cpu_offload: bool,
        model_name: str,
    ) -> FSDP:
        """Load transformer from HF pretrained weights and wrap with FSDP."""
        subfolder = "transformer"
        print(f"⏳ Loading {model_name} from <{pipe_path}>/{subfolder}")
        model = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=pipe_path,
            subfolder=subfolder,
            torch_dtype=model_dtype,
        )
        return fsdp_wrap(
            model=model,
            wrap_modules=wrap_modules,
            param_dtype=model_dtype,
            strategy=fsdp_strategy,
            cpu_offload=fsdp_cpu_offload,
            model_name=model_name,
        )

    def _load_sharded(
        self,
        wrap_modules: tuple,
        *,
        pipe_path: str,
        model_dtype: torch.dtype,
        fsdp_strategy: str,
        fsdp_cpu_offload: bool,
        sharded_weights_dir: str,
        model_name: str,
    ) -> FSDP:
        """Load transformer from pre-converted FSDP shards and wrap with FSDP."""
        if not (sharded_weights_dir and sharded_weights_dir.strip()):
            raise ValueError("❌ use_sharded_weights is true but sharded_weights_dir is empty")
        shard_dir = os.path.join(self.data_and_model_dir, sharded_weights_dir)
        if not os.path.isdir(shard_dir):
            raise ValueError(f"❌ sharded_weights_dir not a directory: {shard_dir}")
        missing = []
        for i in range(self.world_size):
            p = os.path.join(shard_dir, f"flux_shard-{i + 1:05d}-of-{self.world_size:05d}.pt")
            if not os.path.isfile(p):
                missing.append(p)
        if missing:
            raise ValueError(
                f"❌ missing {len(missing)} shard(s) in {shard_dir} (expected {self.world_size}): {missing}"
            )
        shard_path = os.path.join(
            shard_dir, f"flux_shard-{self.rank + 1:05d}-of-{self.world_size:05d}.pt"
        )
        print(f"⏳ Loading {model_name} sharded weights (rank {self.rank}/{self.world_size}) from {shard_path}")

        config = FluxTransformer2DModel.load_config(pipe_path, subfolder="transformer")
        with init_empty_weights():
            model = FluxTransformer2DModel.from_config(config, torch_dtype=model_dtype)

        inferred = _dtype_from_config(pipe_path)
        init_dtype = inferred if inferred is not None else model_dtype

        model = fsdp_wrap(
            model=model,
            wrap_modules=wrap_modules,
            param_dtype=model_dtype,
            strategy=fsdp_strategy,
            cpu_offload=fsdp_cpu_offload,
            from_empty_weights=True,
            init_dtype=init_dtype,
            model_name=model_name,
        )

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            local_sd = torch.load(shard_path, map_location="cpu", weights_only=False)
            set_model_state_dict(
                model,
                model_state_dict=local_sd,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            )

        return model

    def _load_ref_transformer(self, cfg_ref: dict) -> FSDP:
        wrap_modules = (FluxTransformerBlock, FluxSingleTransformerBlock)

        ref_path = cfg_ref["model_path"].strip()
        pipe_path = (
            os.path.join(self.data_and_model_dir, ref_path) if ref_path else self.pipe_path
        )

        # Ref and main transformer share all import params except path/offload.
        ref_dtype = self.model_dtype
        ref_fsdp_strategy = self.fsdp_strategy
        ref_use_sharded_weights = self.use_sharded_weights
        ref_sharded_weights_dir = self.sharded_weights_dir
        ref_offload = cfg_ref["ref_offload"]

        if ref_use_sharded_weights and (self.rank is None or self.world_size is None):
            raise ValueError("❌ env required for ref sharded weights (use_sharded_weights=true)")

        if ref_use_sharded_weights:
            ref_dit = self._load_sharded(
                wrap_modules=wrap_modules,
                pipe_path=pipe_path,
                model_dtype=ref_dtype,
                fsdp_strategy=ref_fsdp_strategy,
                fsdp_cpu_offload=ref_offload,
                sharded_weights_dir=ref_sharded_weights_dir,
                model_name="ref_transformer",
            )
        else:
            ref_dit = self._load_pretrained(
                wrap_modules=wrap_modules,
                pipe_path=pipe_path,
                model_dtype=ref_dtype,
                fsdp_strategy=ref_fsdp_strategy,
                fsdp_cpu_offload=ref_offload,
                model_name="ref_transformer",
            )

        ref_dit.eval()
        ref_dit.requires_grad_(False)
        print("✅ Prepared ref transformer (eval mode)")
        return ref_dit

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
    ) -> tuple[FSDP, FSDP, CLIPTokenizer, T5TokenizerFast]:
        """Load text encoder module (FSDP-wrapped; CPUOffload via FSDP)."""
        subfolder = "tokenizer"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer = CLIPTokenizer.from_pretrained(self.pipe_path, subfolder=subfolder)

        subfolder = "tokenizer_2"
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder}")
        tokenizer_2 = T5TokenizerFast.from_pretrained(self.pipe_path, subfolder=subfolder)
        
        if self.use_sharded_text_encoder:
            clip_shard_dir = os.path.join(self.data_and_model_dir, self.clip_fsdp_shards_dir)
            t5_shard_dir = os.path.join(self.data_and_model_dir, self.t5_fsdp_shards_dir)
            text_encoder = self._load_text_encoder_from_shards(
                CLIPTextModel, CLIPEncoderLayer, "text_encoder",
                clip_shard_dir, "clip_shard"
            )
            text_encoder_2 = self._load_text_encoder_from_shards(
                T5EncoderModel, T5Block, "text_encoder_2",
                t5_shard_dir, "t5_shard"
            )
        else:
            text_encoder = self._load_text_encoder_no_shard(CLIPTextModel, "text_encoder")
            text_encoder_2 = self._load_text_encoder_no_shard(T5EncoderModel, "text_encoder_2")
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        print("✅ Prepared text encoder: CLIPTextModel & T5EncoderModel (eval mode)")
        return text_encoder, text_encoder_2, tokenizer, tokenizer_2

    def _load_text_encoder_no_shard(self, model_cls: type, subfolder: str) -> FSDP:
        """Full weights per rank, NO_SHARD FSDP; text_encoder_offload -> CPUOffload."""
        dtype = _dtype_from_config(self.pipe_path, subfolder) or self.model_dtype
        print(f"⏳ Loading FLUX {subfolder} from <{self.pipe_path}>/{subfolder} (NO_SHARD)")
        model = model_cls.from_pretrained(
            self.pipe_path, subfolder=subfolder, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.eval()
        model.requires_grad_(False)
        return fsdp_wrap(
            model=model,
            wrap_modules=(model_cls,),
            param_dtype=dtype,
            strategy="no_shard",
            cpu_offload=self.text_encoder_offload,
            model_name=model_cls.__name__,
        )

    def _load_text_encoder_from_shards(
        self,
        model_cls: type,
        wrap_cls: type,
        subfolder: str,
        shard_dir: str,
        shard_prefix: str,
    ) -> FSDP:
        """FULL_SHARD; load from pre-converted shards (same as transformer _load_sharded)."""
        if not (shard_dir and shard_dir.strip()):
            raise ValueError(
                f"❌ use_sharded_text_encoder is true but {shard_prefix}_fsdp_shards_dir is empty"
            )
        if not os.path.isdir(shard_dir):
            raise ValueError(f"❌ Shard dir not found: {shard_dir}")
        missing = []
        for i in range(self.world_size):
            p = os.path.join(
                shard_dir, f"{shard_prefix}-{i + 1:05d}-of-{self.world_size:05d}.pt"
            )
            if not os.path.isfile(p):
                missing.append(p)
        if missing:
            raise ValueError(
                f"❌ missing {len(missing)} shard(s) in {shard_dir} "
                f"(expected {self.world_size}): {missing}"
            )
        shard_path = os.path.join(
            shard_dir, f"{shard_prefix}-{self.rank + 1:05d}-of-{self.world_size:05d}.pt"
        )
        print(
            f"⏳ Loading FLUX {subfolder} shards (rank {self.rank}/{self.world_size}) "
            f"from {shard_path}"
        )
        dtype = _dtype_from_config(self.pipe_path, subfolder) or self.model_dtype
        config = model_cls.config_class.from_pretrained(self.pipe_path, subfolder=subfolder)
        with init_empty_weights():
            model = model_cls(config)
        model.eval()
        model.requires_grad_(False)
        model = fsdp_wrap(
            model=model,
            wrap_modules=(wrap_cls,),
            param_dtype=dtype,
            strategy="full_shard",
            cpu_offload=self.text_encoder_offload,
            from_empty_weights=True,
            init_dtype=dtype,
            model_name=model_cls.__name__,
        )
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            local_sd = torch.load(shard_path, map_location="cpu", weights_only=False)
            set_model_state_dict(
                model,
                model_state_dict=local_sd,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            )
        return model

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

        text_embed = {
            "prompt_embed_t5": prompt_embeds,
            "prompt_embed_clip": pooled_prompt_embeds,
            "text_id": text_ids,
        }

        torch.cuda.empty_cache()

        return text_embed
