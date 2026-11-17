"""
MiniMax-H3-Image model: the text-to-image DiT with LoRA, its frozen VAE and text
encoder, and the six-step PDD schedule.

The DiT predicts a data-ward velocity, so x0 = x + sigma * v_native. The solvers of
FDFO and TwinTilt use x0 = x - sigma * v, so velocity returns -v_native.

H3-Image needs a newer diffusers than the rest of the repo, so every module of it
loads inside the method that needs it. See scripts/tools/h3_overlay.sh for the
overlay that provides it.

References:
    - MiniMax-H3-Image:
        - Weights: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private)
            - src/nexus_align/twintilt_h3/native.py: load_schedule, build_layout,
              inject_lora, predict_velocity
            - src/nexus_align/twintilt_h3/sampling.py: unpack_rows, decode_rows
            - src/nexus_align/twintilt_h3/conditioning.py: prepare_conditioning
"""

import os
import json

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoTokenizer

from nexus_align.engine.fsdp import fsdp_wrap
from nexus_align.models.base_model import BaseModel

TEXT_ENCODER_LAYER = 50  # H3 conditions on hidden_states[50], pre-norm
PIXEL_MEAN = (0.485, 0.456, 0.406)  # the ImageNet statistics of the VAE output
PIXEL_STD = (0.229, 0.224, 0.225)


class MiniMaxH3ImageModel(BaseModel):
    """
    FSDP-wrapped MiniMax-H3-Image DiT with LoRA, and the frozen components.

    - self.model / self.ema: the DiT (bf16, with its own fp32 modules), trainable only
      through LoRA. The EMA tracks only the LoRA weights, as in TwinTilt.
    - self.text_encoder: Qwen3-VL cut to 51 layers, sharded like the DiT.
    - self.vae: frozen on the GPU; it decodes one image at a time, as H3 does.
    - self.sigmas: the six-step PDD grid, and self.timesteps = 1 - sigmas.
    """

    def __init__(
        self, cfg_model, build_keys: tuple[str, ...] = ("model", "ema")
    ) -> None:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        from diffusers import AutoencoderKLMiniMaxH3Image

        self.cfg_model = cfg_model
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.ema = None
        self.fsdp_param_dtype = None
        self.save_trainable_only = True  # checkpoints hold only the LoRA weights
        path = cfg_model.path
        fsdp_configs = dict(cfg_model.fsdp)

        # Build the DiT with LoRA, and its EMA copy. The LoRA initialization uses a
        # fixed seed, so that all ranks start with the same weights.
        self.model = None
        if "model" in build_keys:
            model = self.build_model()
            ema = None
            if cfg_model.get("ema_enabled", False) and "ema" in build_keys:
                # Load the weights again instead of a deep copy. The safetensors files
                # are mapped, so the two models share their pages, while a copy would
                # give every rank a second private copy of the 62 GB of weights.
                ema = self.build_model().requires_grad_(False)
            self.model = fsdp_wrap(
                module=model,
                wrap_modules=self.wrap_modules(),
                model_name="model",
                **fsdp_configs,
            )
            self.model.train()
            if ema is not None:
                self.ema = fsdp_wrap(
                    module=ema,
                    wrap_modules=self.wrap_modules(),
                    model_name="ema",
                    **{**fsdp_configs, "activation_ckpt": False},
                )
                self.ema.eval()

        # Load the text encoder. Only the last hidden state gets the final norm, so
        # hidden_states[50] stays pre-norm.
        config = AutoConfig.from_pretrained(os.path.join(path, "text_encoder"))
        config.text_config.num_hidden_layers = TEXT_ENCODER_LAYER + 1
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(path, "text_encoder"), config=config, dtype=torch.bfloat16
        )
        text_encoder.lm_head = None
        text_encoder.requires_grad_(False).eval()
        text_encoder.model = fsdp_wrap(
            module=text_encoder.model,
            wrap_modules=self.text_encoder_wrap_modules(),
            model_name="text_encoder",
            **{**fsdp_configs, "activation_ckpt": False},
        )
        self.text_encoder = text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.processor = Qwen3VLProcessor.from_pretrained(
            os.path.join(path, "processor")
        )

        # Load the VAE, and read the six-step PDD schedule that the DiT's heads expect.
        self.vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
            os.path.join(path, "vae_diffusers"), torch_dtype=torch.bfloat16
        )
        self.vae.to(self.device).requires_grad_(False).eval()
        self.sigmas, self.timesteps = self.build_schedule(path)
        self.scheduler = None  # H3 reads its schedule from pdd_config.json
        self.latent_channels = int(self.vae.config.latent_channels)
        self.patch_size = tuple(int(x) for x in self.model_config().patch_size)
        self.spatial_ratio = int(self.vae.spatial_compression_ratio)
        print(f"✅ Loaded MiniMax-H3-Image components from <{path}>")

    def model_config(self):
        """Return the config of the DiT under its FSDP and PEFT wrappers."""
        return self.model.base_model.model.config

    def build_model(self):
        from diffusers import MiniMaxH3ImageTransformer2DModel

        transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            self.cfg_model.path, subfolder="transformer_diffusers",
            torch_dtype=torch.bfloat16
        )
        transformer.requires_grad_(False)
        blocks = transformer.transformer_blocks
        if any(block.attn.fused_projections for block in blocks):
            raise ValueError("❌ Fused attention bypasses the to_q/to_k/to_v LoRA")
        lora = self.cfg_model.lora
        lora_config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            init_lora_weights=lora.init,
            # A pattern, because the suffixes of a list would also match the refiner.
            target_modules=lora.target_pattern,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(lora.seed)
            # Keep the LoRA weights in the base bf16, as the H3 code does.
            model = get_peft_model(
                transformer, lora_config, autocast_adapter_dtype=False
            )
        layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
        expected = lora.layers_per_block * len(blocks)
        if len(layers) != expected:
            raise ValueError(
                f"❌ The LoRA pattern matched {len(layers)} layers, not {expected}"
            )
        return model

    def wrap_modules(self) -> tuple:
        from diffusers.models.transformers.transformer_minimax_h3_image import (
            MiniMaxH3ImageTokenRefinerBlock, MiniMaxH3ImageTransformerBlock
        )

        return (MiniMaxH3ImageTransformerBlock, MiniMaxH3ImageTokenRefinerBlock)

    def text_encoder_wrap_modules(self) -> tuple:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer
        )

        return (Qwen3VLTextDecoderLayer,)

    def build_schedule(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read the PDD sigmas, and round them as the released sampler does.

        The sampler takes the fp32 timestep t = 1 - sigma, so sigma comes back from it.
        """
        with open(os.path.join(path, "pdd_config.json"), encoding="utf-8") as f:
            config = json.load(f)
        sigmas = [float(x) for x in config["sigmas"]]
        num_steps = int(config["num_inference_steps"])
        if len(sigmas) != num_steps + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError(f"❌ Expected the {num_steps}-step grid from sigma 1 to 0")
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError(f"❌ The H3 sigmas must decrease strictly: {sigmas}")
        timesteps = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype=torch.float32)
        sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
        return sigmas.to(self.device), (1.0 - sigmas[:-1]).to(self.device)

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Update only the trainable EMA parameters, and not the frozen base weights."""
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            if p.requires_grad:
                ema_p.add_((1 - decay) * (p.data - ema_p))

    # ----------------------------------------
    # Prompts, the packed layout, and the noise.
    # ----------------------------------------
    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_length: int) -> tuple[torch.Tensor, ...]:
        """Encode one raw prompt, without template or special tokens, to (1, L, D)."""
        from diffusers.modular_pipelines.minimax_h3_image.encoders import (
            get_qwen3vl_prompt_embeds
        )
        from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
            MINIMAX_H3_TEXT_TAG
        )

        tokens = self.tokenizer(prompt.strip(), add_special_tokens=False)
        token_ids = tokens["input_ids"]
        if not token_ids:
            raise ValueError(f"❌ Prompt produced no tokens: {prompt}")
        token_ids = token_ids[:max_length]
        embeds = get_qwen3vl_prompt_embeds(
            self.text_encoder, self.processor, token_ids, {}, TEXT_ENCODER_LAYER,
            self.device, torch.bfloat16
        )
        tags = torch.full((len(token_ids),), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        return embeds, tags

    def layout(self, text_token_tags: torch.Tensor, resolution: int) -> tuple:
        """Packed [text | image] layout: positions, tags, row indices, and timesteps."""
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
            build_image_packed_sequence
        )
        from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
            MINIMAX_H3_IMAGE_TAG
        )

        _, patch_h, patch_w = self.patch_size
        if resolution % (self.spatial_ratio * patch_h) or resolution % (
            self.spatial_ratio * patch_w
        ):
            raise ValueError(f"❌ Resolution {resolution} does not fit the H3 grid")
        latent_size = resolution // self.spatial_ratio
        packed = build_image_packed_sequence(
            text_token_tags, [], latent_size, latent_size, self.patch_size,
            MINIMAX_H3_IMAGE_TAG,
        )
        position_ids, token_tags, image_rows, text_rows, num_reference_rows = packed
        if num_reference_rows != 0:
            raise ValueError("❌ A text-to-image layout has no reference rows")
        timestep_rows = torch.zeros(position_ids.shape[0], dtype=torch.long)
        rows = tuple(x.to(self.device) for x in (
            position_ids, token_tags, image_rows, text_rows, timestep_rows
        ))
        return (latent_size, *rows)

    def initial_noise(self, seed: int, latent_size: int) -> torch.Tensor:
        """Draw one image's noise rows from a CPU generator, as the H3 sampler does."""
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
            patchify_video_latents
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (1, self.latent_channels, 1, latent_size, latent_size)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        return patchify_video_latents(noise, self.patch_size).to(self.device)

    # ----------------------------------------
    # The DiT and decoding.
    # ----------------------------------------
    def velocity(self, net, rows, embeds, layout, step_index: int) -> torch.Tensor:
        """
        Predict the velocity of the image rows (B, rows, channels) at a PDD step.

        The DiT predicts a data-ward velocity, so this returns -v_native, the
        convention of the FDFO and TwinTilt solvers.
        """
        _, position_ids, token_tags, image_rows, text_rows, timestep_rows = layout
        native = net(
            hidden_states=rows,
            encoder_hidden_states=embeds.expand(rows.shape[0], -1, -1),
            timestep=self.timesteps[step_index: step_index + 1],
            timestep_indices=timestep_rows,
            token_tags=token_tags,
            position_ids=position_ids,
            image_indices=image_rows,
            text_indices=text_rows,
            pdd_step_index=step_index,
            return_dict=False,
        )[0]
        return -native

    def unpack(self, rows: torch.Tensor, latent_size: int) -> torch.Tensor:
        """Invert the row packing: (N, rows, channels) to (N, channels, H, W)."""
        _, patch_h, patch_w = self.patch_size
        grid = rows.reshape(
            -1, latent_size // patch_h, latent_size // patch_w,
            self.latent_channels, patch_h, patch_w,
        ).permute(0, 3, 1, 4, 2, 5)
        return grid.reshape(-1, self.latent_channels, latent_size, latent_size)

    @torch.no_grad()
    def decode(self, rows: torch.Tensor, latent_size: int) -> torch.Tensor:
        """Decode rows (N, rows, channels) to images (N, 3, H, W) in [0, 1]."""
        latents = self.unpack(rows.to(self.device), latent_size).contiguous()
        config = self.vae.config
        mean = torch.tensor(config.latents_mean, device=self.device).view(1, -1, 1, 1)
        std = torch.tensor(config.latents_std, device=self.device).view(1, -1, 1, 1)
        pixel_mean = torch.tensor(PIXEL_MEAN, device=self.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=self.device).view(1, -1, 1, 1)
        images = []
        for latent in latents.split(1):  # H3 decodes one image at a time
            image = self.vae.decode(latent * std + mean, return_dict=False)[0]
            image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)
            # The released pipeline truncates to 8 bits, so the rewards see them too.
            images.append((image * 255).clamp(0, 255).to(torch.uint8).float() / 255)
        return torch.cat(images)
