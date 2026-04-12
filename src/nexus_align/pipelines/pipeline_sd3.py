"""Stable Diffusion 3 evaluation and training pipelines for image generation."""

import os
import random
import numpy as np
from typing import Callable, Any

import torch
import torch.nn.functional as F
import torch.distributed as dist

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers import SD3Transformer2DModel, AutoencoderKL, StableDiffusion3Pipeline
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.utils.torch_utils import randn_tensor

from nexus_align.core.base_pipeline import BaseTrainPipeline
from nexus_align.pipelines.scheduler_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    RLFlowMatchEulerDiscreteScheduler,
)
from nexus_align.utils.progress import TqdmBar


class SD3InferPipeline:
    """Inference pipeline for Stable Diffusion 3."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        kwargs: dict = {},
    ) -> None:
        self.model_name = "SD3"

        pipeline_path = os.path.join(
            kwargs["common"]["data_and_model_dir"],
            kwargs["model"]["path"],
        )
        safetensors_file = kwargs["model"].get(
            "safetensors_file", "sd3_medium_incl_clips_t5xxlfp16.safetensors"
        )
        safetensors_path = os.path.join(pipeline_path, safetensors_file)

        # Load from single-file safetensors and extract components
        print(f"⏳ Loading {self.model_name} pipeline from <{safetensors_path}>")
        diffusers_pipe = StableDiffusion3Pipeline.from_single_file(
            safetensors_path, torch_dtype=dtype,
        )

        # Build local SD3Pipeline from extracted components
        pipe = SD3Pipeline(
            transformer=diffusers_pipe.transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                diffusers_pipe.scheduler.config
            ),
            vae=diffusers_pipe.vae,
            text_encoder=diffusers_pipe.text_encoder,
            tokenizer=diffusers_pipe.tokenizer,
            text_encoder_2=diffusers_pipe.text_encoder_2,
            tokenizer_2=diffusers_pipe.tokenizer_2,
            text_encoder_3=diffusers_pipe.text_encoder_3,
            tokenizer_3=diffusers_pipe.tokenizer_3,
        )
        del diffusers_pipe
        print(f"✅ Loaded pipeline from <{safetensors_path}>")

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
        self.guidance_scale = kwargs["model"]["eval"]["cfg"]
        self.generator = torch.Generator(device=device).manual_seed(
            kwargs["common"]["seed"]
        )

    def __call__(self, data):
        texts = data["text"]

        result = self.pipe(
            prompt=texts,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_infer_steps,
            max_sequence_length=256,
            generator=self.generator,
        ).images

        return {"image": result}


class SD3TrainPipeline(BaseTrainPipeline):
    """Training pipeline for Stable Diffusion 3."""

    # SD3 uses standard AutoencoderKL with scaling_factor and shift_factor.
    # vae_scale_factor = 2^(len(block_out_channels)-1) = typically 8
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
        self.ref_model = model.ref_model
        self.vae = model.vae
        self.encode_prompt = model.encode_prompt

        # Optimize
        self.model_dtype = model_dtype
        self.amp_dtype = amp_dtype
        self.device = device
        self.grad_accu_step = cfg_algo_train["grad_accu_step"]
        self.train_batch_size = cfg_algo_train["train_batch_size"]

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
            self.sample_cfg = cfg_model_algo["sample_cfg"]
            self.timestep_fraction = cfg_model_algo["timestep_fraction"]
            sample_eta = cfg_model_algo["sample_eta"]
            self.scheduler = RLFlowMatchEulerDiscreteScheduler(sample_eta=sample_eta)

    def _call_transformer(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        model=None,
    ) -> torch.Tensor:
        """Call an SD3 transformer, reusable for both main and ref model."""
        if model is None:
            model = self.model

        with torch.amp.autocast(
            device_type=self.device.type, dtype=self.amp_dtype
        ):
            model_output = model(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            ).sample.to(dtype=self.model_dtype)

        return model_output

    def _apply_cfg(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        model=None,
    ) -> torch.Tensor:
        """Compute noise prediction with Classifier-Free Guidance.

        When sample_cfg <= 1, falls back to a single conditional forward pass.
        """
        if self.sample_cfg <= 1.0:
            return self._call_transformer(
                latents, timesteps, prompt_embeds, pooled_prompt_embeds, model=model,
            )

        cat_latents = torch.cat([latents, latents])
        cat_timesteps = torch.cat([timesteps, timesteps])
        cat_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        cat_pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        cat_output = self._call_transformer(
            cat_latents, cat_timesteps, cat_embeds, cat_pooled, model=model,
        )
        uncond_output, cond_output = cat_output.chunk(2)
        return uncond_output + self.sample_cfg * (cond_output - uncond_output)

    def prepare_data(self, data: dict) -> dict:
        """Prepare data from dataloader batch."""
        prompts = data["text"]
        features = ["prompt_embeds", "pooled_prompt_embeds"]
        if not all(t in data for t in features):
            text_embed = self.encode_prompt(prompts)
            data.update(text_embed)

        if self.algo_name == "grpo":
            neg_features = ["negative_prompt_embeds", "negative_pooled_prompt_embeds"]
            if not all(t in data for t in neg_features):
                neg_embed = self.encode_prompt([""] * len(prompts))
                data["negative_prompt_embeds"] = neg_embed["prompt_embeds"]
                data["negative_pooled_prompt_embeds"] = neg_embed["pooled_prompt_embeds"]

        sampled_prompts = random.sample(prompts, min(4, len(prompts)))
        print("\nSampled training prompts:\n" + "\n".join(sampled_prompts))

        if self.algo_name == "grpo":
            data["keys_to_build_groups"] = {
                "text",
                "prompt_embeds",
                "pooled_prompt_embeds",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
            }

        return data

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        """Rollout: generate responses from the prepared data."""
        # Prepare denoising schedule
        t = torch.linspace(1, 0, self.sample_steps + 1)
        sigma_schedule = (self.sample_shift * t) / (1 + (self.sample_shift - 1) * t)

        # Prepare init latents (SD3 uses standard spatial latents, no packing)
        latent_c = 16
        latent_h = self.sample_height // self.VAE_SCALE_FACTOR
        latent_w = self.sample_width // self.VAE_SCALE_FACTOR
        shared_latents = torch.randn(
            (1, latent_c, latent_h, latent_w), dtype=self.model_dtype
        )

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
            pooled_prompt_embeds = data["pooled_prompt_embeds"][b_idx]
            negative_prompt_embeds = data["negative_prompt_embeds"][b_idx]
            negative_pooled_prompt_embeds = data["negative_pooled_prompt_embeds"][b_idx]

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

                model_output = self._apply_cfg(
                    latents, timesteps, prompt_embeds, pooled_prompt_embeds,
                    negative_prompt_embeds, negative_pooled_prompt_embeds,
                )

                # SD3 uses spatial latents: flatten for step_fn, then reshape
                B, C, H, W = latents.shape
                latents_flat = latents.view(B, -1).unsqueeze(1)
                output_flat = model_output.view(B, -1).unsqueeze(1)

                latents_flat, pred_flat, log_prob = self.scheduler.step_fn(
                    model_output=output_flat,
                    latents=latents_flat,
                    sigmas=sigma_schedule,
                    index=i,
                )

                latents = latents_flat.view(B, C, H, W)
                pred_original = pred_flat.view(B, C, H, W)

                latents_steps.append(latents)
                log_probs_steps.append(log_prob)

            all_latents.append(torch.stack(latents_steps, dim=1))
            all_log_probs.append(torch.stack(log_probs_steps, dim=1))
            texts += [data["text"][t] for t in b_idx]

            # ----------------------------------------
            # Save sampled results
            # ----------------------------------------
            image_processor = VaeImageProcessor(self.VAE_SCALE_FACTOR)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            ):
                scaling_factor = self.vae.config.scaling_factor
                shift_factor = self.vae.config.shift_factor
                pred_decoded = pred_original / scaling_factor + shift_factor
                pred_decoded = pred_decoded.to(
                    device=self.vae.device, dtype=self.vae.dtype
                )
                image = self.vae.decode(pred_decoded, return_dict=False)[0]

                img_pil = image_processor.postprocess(image)

            train_state = "-".join(
                [f"{k}{data[k]}" for k in ["epoch", "step", "total_step"]]
            )
            for i, img in enumerate(img_pil):
                img_num = len(images)
                gs = self.group_size
                img_idx = f"data{img_num // gs}-res{img_num % gs}"
                file_name = f"sd3-{train_state}-rank{rank}-{img_idx}.png"
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

        latents_trimmed = all_latents[:, :-2]
        next_latents_trimmed = all_latents[:, 1:-1]
        timesteps_trimmed = timesteps[:, :-1]

        ref_log_probs = self._compute_ref_log_probs(
            latents=latents_trimmed,
            next_latents=next_latents_trimmed,
            timesteps=timesteps_trimmed,
            sigma_schedule=sigma_schedule,
            prompt_embeds=data["prompt_embeds"],
            pooled_prompt_embeds=data["pooled_prompt_embeds"],
            negative_prompt_embeds=data["negative_prompt_embeds"],
            negative_pooled_prompt_embeds=data["negative_pooled_prompt_embeds"],
        )

        data = {
            "reward_inputs": {"image": images, "image_pil": image_pils, "text": texts},
            "latents": latents_trimmed,
            "next_latents": next_latents_trimmed,
            "timesteps": timesteps_trimmed,
            "log_probs": torch.cat(all_log_probs)[:, :-1],
            "ref_log_probs": ref_log_probs,
            "prompt_embeds": data["prompt_embeds"],
            "pooled_prompt_embeds": data["pooled_prompt_embeds"],
            "negative_prompt_embeds": data["negative_prompt_embeds"],
            "negative_pooled_prompt_embeds": data["negative_pooled_prompt_embeds"],
            "sigma_schedule": sigma_schedule,
        }

        torch.cuda.empty_cache()

        return data

    @torch.no_grad()
    def _compute_ref_log_probs(
        self,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        sigma_schedule: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probs from the frozen reference model for KL divergence."""
        if self.ref_model is None:
            return torch.zeros_like(latents[:, :, 0, 0, 0])

        batch, num_steps = latents.shape[0], latents.shape[1]
        batch_ind = torch.arange(batch).chunk(batch // self.sample_batch_size)

        all_ref_log_probs = []
        bar = TqdmBar(total=len(batch_ind), desc="Computing ref log probs", unit="batch")
        self.ref_model.eval()

        for b_idx in batch_ind:
            ref_log_probs_steps = []
            step_prompt_embeds = prompt_embeds[b_idx]
            step_pooled = pooled_prompt_embeds[b_idx]
            step_neg_embeds = negative_prompt_embeds[b_idx]
            step_neg_pooled = negative_pooled_prompt_embeds[b_idx]

            for i in range(num_steps):
                step_latents = latents[b_idx, i].to(self.device)
                step_next_latents = next_latents[b_idx, i].to(self.device)
                step_timesteps = timesteps[b_idx, i].to(self.device)

                model_output = self._apply_cfg(
                    step_latents, step_timesteps,
                    step_prompt_embeds, step_pooled,
                    step_neg_embeds, step_neg_pooled,
                    model=self.ref_model,
                )

                B, C, H, W = step_latents.shape
                latents_flat = step_latents.view(B, -1).unsqueeze(1)
                output_flat = model_output.view(B, -1).unsqueeze(1)
                next_flat = step_next_latents.view(B, -1).unsqueeze(1)

                _, _, ref_log_prob = self.scheduler.step_fn(
                    model_output=output_flat,
                    latents=latents_flat,
                    sigmas=sigma_schedule,
                    index=i,
                    prev_sample=next_flat,
                )
                ref_log_probs_steps.append(ref_log_prob)

            all_ref_log_probs.append(torch.stack(ref_log_probs_steps, dim=1))
            bar.update(1)

        bar.close()
        return torch.cat(all_ref_log_probs, dim=0)

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
        for key in ["timesteps", "latents", "next_latents", "log_probs", "ref_log_probs"]:
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
                        pooled_prompt_embeds = responses["pooled_prompt_embeds"][
                            s_idx:e_idx
                        ]
                        neg_embeds = responses["negative_prompt_embeds"][s_idx:e_idx]
                        neg_pooled = responses["negative_pooled_prompt_embeds"][s_idx:e_idx]
                        latents = responses["latents"][s_idx:e_idx, step]

                        model_output = self._apply_cfg(
                            latents,
                            responses["timesteps"][s_idx:e_idx, step],
                            prompt_embeds,
                            pooled_prompt_embeds,
                            neg_embeds,
                            neg_pooled,
                        )

                        # Flatten spatial dims for step_fn
                        B, C, H, W = latents.shape
                        latents_flat = latents.view(B, -1).unsqueeze(1)
                        output_flat = model_output.view(B, -1).unsqueeze(1)
                        next_flat = responses["next_latents"][s_idx:e_idx, step]
                        next_flat = next_flat.view(B, -1).unsqueeze(1)

                        _, _, new_log_probs = self.scheduler.step_fn(
                            model_output=output_flat,
                            latents=latents_flat,
                            sigmas=responses["sigma_schedule"],
                            index=p[s_idx:e_idx, step],
                            prev_sample=next_flat,
                        )
                        return new_log_probs

                    return forward_fn

                ref_log_probs = responses["ref_log_probs"][start_idx:end_idx, t]

                yield {
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "advantages": advantages_slice,
                    "forward_fn": make_forward_fn(),
                    "backward_scale": 1.0 / (self.grad_accu_step * train_timesteps),
                    "should_optimizer_step": should_optimizer_step,
                    "group_idx": idx,
                    "progress_info": progress_info,
                }


# --------------------------------------------------------------------------------
# SD3 Pipeline
# --------------------------------------------------------------------------------
# A simplified version of the diffusers implementation for easy customization.
# NOTE: Some checks have been removed for simplicity, which may increase risk.
# --------------------------------------------------------------------------------
# Usage:
#     from nexus_align.pipelines.pipeline_sd3 import SD3Pipeline
#     (official) from diffusers import StableDiffusion3Pipeline
# --------------------------------------------------------------------------------
class SD3Pipeline(
    DiffusionPipeline,
    SD3LoraLoaderMixin,
    FromSingleFileMixin,
):
    """
    A simplified Stable Diffusion 3 pipeline for text-to-image generation.

    Reference:
        SD3: https://arxiv.org/abs/2403.03206
        Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
    """

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    )
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 2
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.FloatTensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    max_sequence_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        clip_skip: int | None = None,
        clip_model_index: int = 0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        negative_prompt_3: str | list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        clip_skip: int | None = None,
        max_sequence_length: int = 256,
        lora_scale: float | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = device or self._execution_device

        # Set LoRA scale
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt, device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip, clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2, device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip, clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = F.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embed, pooled_prompt_2_embed], dim=-1
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2]
                if isinstance(negative_prompt_2, str)
                else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3]
                if isinstance(negative_prompt_3, str)
                else negative_prompt_3
            )

            negative_prompt_embed, negative_pooled_prompt_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt, device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None, clip_model_index=0,
                )
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt_2, device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None, clip_model_index=1,
                )
            )
            negative_clip_prompt_embeds = torch.cat(
                [negative_prompt_embed, negative_prompt_2_embed], dim=-1
            )

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = F.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat(
                [negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2
            )
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

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
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 7.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        negative_prompt_3: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict | None = None,
        clip_skip: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 256,
    ) -> StableDiffusion3PipelineOutput | tuple:
        # --------------------------------------------------------------------------------
        # Prepare
        # --------------------------------------------------------------------------------
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        # --------------------------------------------------------------------------------
        # Encode prompt
        # --------------------------------------------------------------------------------
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # --------------------------------------------------------------------------------
        # Prepare latents
        # --------------------------------------------------------------------------------
        num_channels_latents = self.transformer.config.in_channels
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

        # --------------------------------------------------------------------------------
        # Prepare timesteps
        # --------------------------------------------------------------------------------
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_dynamic_shifting")
            and self.scheduler.config.use_dynamic_shifting
        ):
            _, _, h, w = latents.shape
            image_seq_len = (h // self.patch_size) * (w // self.patch_size)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            self.scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
            )
        else:
            self.scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
            )
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # --------------------------------------------------------------------------------
        # Denoising loop
        # --------------------------------------------------------------------------------
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

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
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


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
