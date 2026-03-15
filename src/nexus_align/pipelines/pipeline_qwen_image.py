"""Qwen-Image evaluation and training pipelines for image generation."""

import os
import random
import numpy as np
from typing import Callable, Any

import torch
import torch.distributed as dist

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.utils.torch_utils import randn_tensor

from nexus_align.core.base_pipeline import BaseTrainPipeline
from nexus_align.pipelines.scheduler_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    RLFlowMatchEulerDiscreteScheduler,
)
from nexus_align.utils.progress import TqdmBar


class QwenImageInferPipeline:
    """Inference pipeline for Qwen-Image."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        kwargs: dict = {},
    ) -> None:
        self.model_name = "QwenImage"

        pipeline_path = os.path.join(
            kwargs["common"]["data_and_model_dir"],
            kwargs["model"]["path"],
        )

        print(f"⏳ Loading {self.model_name} transformer from <{pipeline_path}>/transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            pipeline_path, subfolder="transformer"
        )

        print(f"⏳ Loading {self.model_name} vae from <{pipeline_path}>/vae")
        vae = AutoencoderKLQwenImage.from_pretrained(
            pipeline_path, subfolder="vae"
        )

        print(f"⏳ Loading {self.model_name} text_encoder from <{pipeline_path}>/text_encoder")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pipeline_path, subfolder="text_encoder"
        )

        print(f"⏳ Loading {self.model_name} tokenizer from <{pipeline_path}>/tokenizer")
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pipeline_path, subfolder="tokenizer"
        )

        print(f"⏳ Loading {self.model_name} scheduler from <{pipeline_path}>/scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pipeline_path, subfolder="scheduler"
        )

        pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        print(f"✅ Loaded pipeline from <{pipeline_path}>")

        # Load checkpoint
        ckpt_path = kwargs["model"]["eval"].get("ckpt_path", None)
        if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):
            print(f"⏳ Loading {self.model_name} checkpoint from <{ckpt_path}>")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            pipe.transformer.load_state_dict(state_dict)

        self.pipe = pipe.to(dtype=dtype, device=device)
        self.height = kwargs["model"]["eval"]["height"]
        self.width = kwargs["model"]["eval"]["width"]
        self.num_infer_steps = kwargs["model"]["eval"]["num_infer_steps"]
        self.true_cfg_scale = kwargs["model"]["eval"].get("true_cfg_scale", 1.0)
        self.generator = torch.Generator(device=device).manual_seed(
            kwargs["common"]["seed"]
        )

    def __call__(self, data):
        texts = data["text"]

        result = self.pipe(
            prompt=texts,
            height=self.height,
            width=self.width,
            true_cfg_scale=self.true_cfg_scale,
            num_inference_steps=self.num_infer_steps,
            max_sequence_length=512,
            generator=self.generator,
        ).images

        return {"image": result}


class QwenImageTrainPipeline(BaseTrainPipeline):
    """Training pipeline for Qwen-Image."""

    # Qwen-Image VAE uses learned mean/std instead of scaling_factor/shift_factor.
    # vae_scale_factor = 2^len(temperal_downsample) = 2^3 = 8
    VAE_SCALE_FACTOR = 8

    def __init__(
        self,
        model: Any,
        device: torch.device,
        model_dtype: torch.dtype,
        amp_dtype: torch.dtype,
        kwargs: dict = {},
    ) -> None:
        cfg_algo_train = kwargs["algorithm"].get("train", {})
        cfg_algo_run = kwargs["algorithm"].get("run", {})
        cfg_model_algo = kwargs["model"].get(kwargs["algorithm"]["name"], {})

        # Model
        self.model = model.model
        self.vae = model.vae
        self.encode_prompt = model.encode_prompt

        # Optimize
        self.model_dtype = model_dtype
        self.amp_dtype = amp_dtype
        self.device = device
        self.grad_accu_step = cfg_algo_train["grad_accu_step"]

        # Algorithm-specific
        self.algo_name = kwargs["algorithm"]["name"]
        if self.algo_name == "grpo":
            self.sample_batch_size = cfg_algo_run["sample_batch_size"]
            self.group_size = cfg_algo_run["group_size"]
            self.sample_save_dir = cfg_algo_run["sample_save_dir"]
            os.makedirs(self.sample_save_dir, exist_ok=True)
            self.sample_height = cfg_model_algo["sample_height"]
            self.sample_width = cfg_model_algo["sample_width"]
            self.sample_shift = cfg_model_algo["sample_shift"]
            self.sample_steps = cfg_model_algo["sample_steps"]
            self.timestep_fraction = cfg_model_algo["timestep_fraction"]
            sample_eta = cfg_model_algo["sample_eta"]
            self.scheduler = RLFlowMatchEulerDiscreteScheduler(sample_eta=sample_eta)

    def prepare_data(self, data: dict) -> dict:
        """Prepare data from dataloader batch."""
        prompts = data["text"]
        features = ["prompt_embeds"]
        if not all(t in data for t in features):
            text_embed = self.encode_prompt(prompts)
            data.update(text_embed)

        sampled_prompts = random.sample(prompts, min(4, len(prompts)))
        print("\nSampled training prompts:\n" + "\n".join(sampled_prompts))

        if self.algo_name == "grpo":
            keys = {"text", "prompt_embeds"}
            if data.get("prompt_embeds_mask") is not None:
                keys.add("prompt_embeds_mask")
            data["keys_to_build_groups"] = keys

        return data

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        """Rollout: generate responses from the prepared data."""
        # Prepare denoising schedule
        t = torch.linspace(1, 0, self.sample_steps + 1)
        sigma_schedule = (self.sample_shift * t) / (1 + (self.sample_shift - 1) * t)

        # Prepare init latents with noise
        # Qwen-Image uses z_dim=16 latent channels, same packed dim as FLUX
        latent_c = 16
        latent_h = self.sample_height // self.VAE_SCALE_FACTOR
        latent_w = self.sample_width // self.VAE_SCALE_FACTOR
        shared_latents = torch.randn(
            (1, latent_c, latent_h, latent_w), dtype=self.model_dtype
        )
        shared_latents = _pack_latents(shared_latents, latent_c, latent_h, latent_w)

        # img_shapes for RoPE: (frame=1, h_patches, w_patches) per batch item
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]

        # Prepare sampling
        rank = dist.get_rank()
        batch = len(data["text"])
        batch_ind = torch.arange(batch).chunk(batch // self.sample_batch_size)

        # Run sampling
        all_latents, all_log_probs = list(), list()
        images, image_pils, texts = list(), list(), list()
        bar = TqdmBar(total=len(batch_ind), desc="🚀 Sampling responses", unit="batch")
        self.model.eval()
        for _, b_idx in enumerate(batch_ind):
            batch_size = len(b_idx)
            latents = torch.cat([shared_latents] * batch_size, dim=0).to(self.device)
            prompt_embeds = data["prompt_embeds"][b_idx]
            prompt_embeds_mask = data.get("prompt_embeds_mask")
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask[b_idx]

            batch_img_shapes = img_shapes * batch_size

            # ----------------------------------------
            # Run denoising
            # ----------------------------------------
            latents_steps = [latents]
            log_probs_steps = []
            for i in range(self.sample_steps):
                timestep = int(sigma_schedule[i] * 1000)
                timesteps = torch.full(
                    [batch_size], timestep, device=self.device, dtype=torch.long
                )

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.amp_dtype
                ):
                    model_output = self.model(
                        hidden_states=latents,
                        timestep=timesteps / 1000,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        img_shapes=batch_img_shapes,
                    ).sample.to(dtype=self.model_dtype)

                latents, pred_original, log_prob = self.scheduler.step_fn(
                    model_output=model_output,
                    latents=latents,
                    sigmas=sigma_schedule,
                    index=i,
                )

                latents_steps.append(latents)
                log_probs_steps.append(log_prob)

            all_latents.append(torch.stack(latents_steps, dim=1))
            all_log_probs.append(torch.stack(log_probs_steps, dim=1))
            texts += [data["text"][t] for t in b_idx]

            # ----------------------------------------
            # Save sampled results
            # ----------------------------------------
            self.vae.enable_tiling()
            image_processor = VaeImageProcessor(self.VAE_SCALE_FACTOR * 2)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            ):
                pred_unpacked = _unpack_latents(
                    latents=pred_original,
                    height=self.sample_height,
                    width=self.sample_width,
                    vae_scale_factor=self.VAE_SCALE_FACTOR,
                )
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(pred_unpacked.device, pred_unpacked.dtype)
                )
                latents_std = (
                    1.0
                    / torch.tensor(self.vae.config.latents_std)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(pred_unpacked.device, pred_unpacked.dtype)
                )
                pred_unpacked = pred_unpacked / latents_std + latents_mean
                pred_unpacked = pred_unpacked.to(
                    device=self.vae.device, dtype=self.vae.dtype
                )
                image = self.vae.decode(pred_unpacked, return_dict=False)[0]

                if image.ndim == 5:
                    image = image.squeeze(2)
                img_pil = image_processor.postprocess(image)

            train_state = "-".join(
                [f"{k}{data[k]}" for k in ["epoch", "step", "total_step"]]
            )
            for i, img in enumerate(img_pil):
                img_num = len(images)
                gs = self.group_size
                img_idx = f"data{img_num // gs}-res{img_num % gs}"
                file_name = f"qwen_image-{train_state}-rank{rank}-{img_idx}.png"
                save_path = os.path.join(self.sample_save_dir, file_name)
                try:
                    img.save(save_path)
                except Exception as e:
                    print(f"❌ Error saving sampled result to {file_name}: {e}")
                    continue
                images.append(save_path)
                image_pils.append(img)

            bar.update(1)
        bar.close()

        all_latents = torch.cat(all_latents)
        timestep_values = [
            int(sigma * 1000) for sigma in sigma_schedule[: self.sample_steps]
        ]
        timesteps = torch.tensor([timestep_values] * batch, dtype=torch.int64)

        shared_img_shapes = img_shapes

        data = {
            "reward_inputs": {"image": images, "image_pil": image_pils, "text": texts},
            "latents": all_latents[:, :-2],
            "next_latents": all_latents[:, 1:-1],
            "timesteps": timesteps[:, :-1],
            "log_probs": torch.cat(all_log_probs)[:, :-1],
            "img_shapes": shared_img_shapes,
            "prompt_embeds": data["prompt_embeds"],
            "prompt_embeds_mask": data.get("prompt_embeds_mask"),
            "sigma_schedule": sigma_schedule,
        }

        torch.cuda.empty_cache()

        return data

    def iterate_training_items(self, responses: dict):
        """
        Yield training items, called by train_one_step() in algorithms.

        Return:
          - item (`dict`):
            - old_log_probs (`torch.Tensor`): log probabilities of the previous timestep.
            - advantages (`torch.Tensor`): raw advantages, algorithm will clamp.
            - forward_fn (`callable`): callable that returns new log probabilities.
            - backward_scale (`float`): scale for backward pass.
            - should_optimizer_step (`bool`): whether to update the optimizer.
            - group_idx (`int`): index of the group.
            - progress_info (`dict` | None): dict with "total", "desc", "unit" for progress bar.
        """
        # Shuffle timesteps and train on a random fraction
        # Reference: DanceGRPO (https://arxiv.org/pdf/2505.07818)
        batch = len(responses["timesteps"])
        step_len = len(responses["timesteps"][0])
        perms = torch.stack([torch.randperm(step_len) for _ in range(batch)])
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            ran = torch.arange(batch)
            responses[key] = responses[key][ran[:, None], perms]
        all_timesteps = len(responses["timesteps"][0])
        train_timesteps = int(all_timesteps * self.timestep_fraction)

        groups = batch // self.group_size
        assert groups % self.grad_accu_step == 0

        for idx in range(groups):
            start_idx = idx * self.group_size
            end_idx = (idx + 1) * self.group_size
            advantages_slice = responses["advantages"][start_idx:end_idx]

            perc = int(self.timestep_fraction * 100)
            progress_info = {
                "total": train_timesteps,
                "desc": f"🚀 Training on random {train_timesteps}/{all_timesteps} steps ({perc}%)",
                "unit": "step",
            }

            for t in range(train_timesteps):
                old_log_probs = responses["log_probs"][start_idx:end_idx, t]

                should_update_group = (idx + 1) % self.grad_accu_step == 0
                should_step_group = t == train_timesteps - 1
                should_optimizer_step = should_update_group and should_step_group

                def make_forward_fn(
                    s_idx=start_idx, e_idx=end_idx, step=t, p=perms
                ):
                    def forward_fn():
                        prompt_embeds = responses["prompt_embeds"][s_idx:e_idx]
                        prompt_embeds_mask = responses.get("prompt_embeds_mask")
                        if prompt_embeds_mask is not None:
                            prompt_embeds_mask = prompt_embeds_mask[s_idx:e_idx]
                        latents = responses["latents"][s_idx:e_idx, step]

                        slice_size = e_idx - s_idx
                        batch_img_shapes = responses["img_shapes"] * slice_size

                        with torch.amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        ):
                            model_output = self.model(
                                hidden_states=latents,
                                timestep=responses["timesteps"][s_idx:e_idx, step]
                                / 1000,
                                encoder_hidden_states=prompt_embeds,
                                encoder_hidden_states_mask=prompt_embeds_mask,
                                img_shapes=batch_img_shapes,
                            ).sample.to(dtype=self.model_dtype)

                        _, _, new_log_probs = self.scheduler.step_fn(
                            model_output=model_output,
                            latents=latents,
                            sigmas=responses["sigma_schedule"],
                            index=p[s_idx:e_idx, step],
                            prev_sample=responses["next_latents"][s_idx:e_idx, step],
                        )
                        return new_log_probs

                    return forward_fn

                yield {
                    "old_log_probs": old_log_probs,
                    "advantages": advantages_slice,
                    "forward_fn": make_forward_fn(),
                    "backward_scale": 1.0 / (self.grad_accu_step * train_timesteps),
                    "should_optimizer_step": should_optimizer_step,
                    "group_idx": idx,
                    "progress_info": progress_info,
                }


# --------------------------------------------------------------------------------
# Qwen-Image Pipeline
# --------------------------------------------------------------------------------
# A simplified version of the diffusers implementation for easy customization.
# NOTE: Some checks have been removed for simplicity, which may increase risk.
# --------------------------------------------------------------------------------
# Usage:
#     from nexus_align.pipelines.pipeline_qwen_image import QwenImagePipeline
#     (official) from diffusers import QwenImagePipeline
# --------------------------------------------------------------------------------
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute mu for dynamic shifting in FlowMatch scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def _pack_latents(
    latents: torch.Tensor,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack latent from (B, C, H, W) or (B, 1, C, H, W) to (B, H/2*W/2, C*4)."""
    batch_size = latents.shape[0]
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    return latents


def _unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    """Unpack latent from (B, H/2*W/2, C*4) to (B, C, 1, H, W) with temporal dim."""
    batch_size, _, channels = latents.shape

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(
        batch_size, height // 2, width // 2, channels // 4, 2, 2
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents


class QwenImagePipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    """
    A simplified Qwen-Image pipeline for text-to-image generation.

    Reference:
        Qwen-Image: https://huggingface.co/Qwen/Qwen-Image
        Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

    def _extract_masked_hidden(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max(e.size(0) for e in split_hidden_states)

        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ) -> tuple[torch.FloatTensor, torch.LongTensor | None]:
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, device
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(
                1, num_images_per_prompt, 1
            )
            prompt_embeds_mask = prompt_embeds_mask.view(
                batch_size * num_images_per_prompt, seq_len
            )
            if prompt_embeds_mask.all():
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: torch.Tensor = None,
    ) -> torch.Tensor:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )
        latents = _pack_latents(latents, num_channels_latents, height, width)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ) -> QwenImagePipelineOutput | tuple:
        # --------------------------------------------------------------------------------
        # Prepare
        # --------------------------------------------------------------------------------
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # --------------------------------------------------------------------------------
        # Encode prompt
        # --------------------------------------------------------------------------------
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # --------------------------------------------------------------------------------
        # Prepare latents
        # --------------------------------------------------------------------------------
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [[
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
        ]] * batch_size

        # --------------------------------------------------------------------------------
        # Prepare timesteps
        # --------------------------------------------------------------------------------
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # Handle guidance embedding
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError(
                "guidance_scale is required for guidance-distilled model."
            )
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # --------------------------------------------------------------------------------
        # Denoising loop
        # --------------------------------------------------------------------------------
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = _unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                1.0
                / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
