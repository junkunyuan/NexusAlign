"""FLUX evaluation and training pipelines for image generation."""

import os
import random
import numpy as np
from typing import Callable, Any

import torch
import torch.distributed as dist

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.loaders import (
    FluxIPAdapterMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
)

from nexus_align.core.base_pipeline import BaseTrainPipeline
from nexus_align.pipelines.scheduler_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    RLFlowMatchEulerDiscreteScheduler,
)
from nexus_align.utils.progress import TqdmBar


class FluxInferPipeline:
    """Inference pipeline for FLUX."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        kwargs: dict = {},
    ) -> None:
        self.model_name = "FLUX"
        
        pipeline_path = os.path.join(
            kwargs["common"]["data_and_model_dir"], 
            kwargs["model"]["path"]
        )
        print(f"⏳ Loading {self.model_name} VAE from <{pipeline_path}/vae>")
        vae = AutoencoderKL.from_pretrained(pipeline_path, subfolder="vae")

        print(f"⏳ Loading {self.model_name} text_encoder from <{pipeline_path}/text_encoder>")
        text_encoder = CLIPTextModel.from_pretrained(
            pipeline_path, subfolder="text_encoder"
        )
        
        print(f"⏳ Loading {self.model_name} text_encoder_2 from <{pipeline_path}/text_encoder_2>")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            pipeline_path, subfolder="text_encoder_2"
        )
        
        print(f"⏳ Loading {self.model_name} tokenizer from <{pipeline_path}/tokenizer>")
        tokenizer = CLIPTokenizer.from_pretrained(pipeline_path, subfolder="tokenizer")
        
        print(f"⏳ Loading {self.model_name} tokenizer 2 from <{pipeline_path}/tokenizer_2>")
        tokenizer_2 = T5TokenizerFast.from_pretrained(pipeline_path, subfolder="tokenizer_2")

        print(f"⏳ Loading {self.model_name} transformer from <{pipeline_path}/transformer>")
        transformer = FluxTransformer2DModel.from_pretrained(pipeline_path, subfolder="transformer")
        
        print(f"⏳ Loading {self.model_name} scheduler from <{pipeline_path}/scheduler>")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pipeline_path, subfolder="scheduler")

        pipe = FluxPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
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
            max_sequence_length=512,
            generator=self.generator,
        ).images

        return {"image": result}


class FluxTrainPipeline(BaseTrainPipeline):
    """Training pipeline for FLUX."""

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
            self.sample_cfg = torch.tensor(
                [cfg_model_algo["sample_cfg"]],
                dtype=model_dtype,
                device=device,
            )
            self.timestep_fraction = cfg_model_algo["timestep_fraction"]
            sample_eta = cfg_model_algo["sample_eta"]
            self.scheduler = RLFlowMatchEulerDiscreteScheduler(sample_eta=sample_eta)

    def prepare_data(self, data: dict) -> dict:
        """Prepare data from dataloader batch."""
        # Load data
        prompts = data["text"]
        features = ["prompt_embed_t5", "prompt_embed_clip", "text_id"]
        if not all(t in data for t in features):
            text_embed = self.encode_prompt(prompts)
            data.update(text_embed)

        # Print sampled prompts for debugging
        sampled_prompts = random.sample(prompts, min(4, len(prompts)))
        print("\nSampled training prompts:\n" + "\n".join(sampled_prompts))
        
        if self.algo_name == "grpo":
            data["keys_to_build_groups"] = {
                "text",
                "prompt_embed_t5", 
                "prompt_embed_clip"
            }

        return data

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        """Rollout: generate responses from the prepared data."""
        # Prepare denoising schedule
        t = torch.linspace(1, 0, self.sample_steps + 1)
        sigma_schedule = (self.sample_shift * t) / (1 + (self.sample_shift - 1) * t)

        # Prepare init latents with noise
        latent_c = 16
        latent_h = self.sample_height // 8
        latent_w = self.sample_width // 8
        shared_latents = torch.randn((1, latent_c, latent_h, latent_w), dtype=self.model_dtype)
        shared_latents = pack_latents(shared_latents, latent_c, latent_h, latent_w)

        # Prepare sampling
        rank = dist.get_rank()
        batch = len(data["text"])
        batch_ind = torch.arange(batch).chunk(batch // self.sample_batch_size)
        shared_image_id = prepare_latent_image_ids(latent_h // 2, latent_w // 2).to(self.device)

        # Run sampling
        all_latents, all_log_probs = list(), list()
        images, image_pils, texts = list(), list(), list()
        bar = TqdmBar(total=len(batch_ind), desc="🚀 Sampling responses", unit="batch")
        self.model.eval()
        for _, b_idx in enumerate(batch_ind):
            batch_size = len(b_idx)
            latents = torch.cat([shared_latents] * batch_size, dim=0).to(self.device)  # [b, l, 64]
            prompt_embed_t5 = data["prompt_embed_t5"][b_idx]  # [b, 512, 4096]
            prompt_embed_clip = data["prompt_embed_clip"][b_idx]  # [b, 768]
            text_id = data["text_id"]  # [512, 3]
            image_id = shared_image_id  # [l, 3]

            # ----------------------------------------
            # Run denoising
            # ----------------------------------------
            latents_steps = [latents]
            log_probs_steps = []
            for i in range(self.sample_steps):
                timestep = int(sigma_schedule[i] * 1000)
                timesteps = torch.full([batch_size], timestep, device=self.device, dtype=torch.long)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    model_output = self.model(
                        hidden_states=latents,
                        timestep=timesteps / 1000,
                        encoder_hidden_states=prompt_embed_t5,
                        pooled_projections=prompt_embed_clip,
                        txt_ids=text_id,
                        img_ids=image_id,
                        guidance=self.sample_cfg,
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
            image_processor = VaeImageProcessor(16)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                pred_original = unpack_latents(
                    latents=pred_original, 
                    height=self.sample_height, 
                    width=self.sample_width, 
                    vae_scale_factor=8
                )
                # VAE decode
                scaling_factor = self.vae.config.scaling_factor
                shift_factor = self.vae.config.shift_factor
                pred_original = pred_original / scaling_factor + shift_factor
                pred_original = pred_original.to(
                    device=self.vae.device, 
                    dtype=self.vae.dtype
                )
                image = self.vae.decode(pred_original, return_dict=False)[0]
                
                img_pil = image_processor.postprocess(image)
            
            train_state = "-".join([f"{k}{data[k]}" for k in ["epoch", "step", "total_step"]])
            for i, img in enumerate(img_pil):
                img_num = len(images)
                gs = self.group_size
                img_idx = f"data{img_num // gs}-res{img_num % gs}"
                file_name = f"flux-{train_state}-rank{rank}-{img_idx}.png"
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

        data = {
            "reward_inputs": {"image": images, "image_pil": image_pils, "text": texts},
            "latents": all_latents[:, :-2],  # [b, sample_steps - 1, l, 64]
            "next_latents": all_latents[:, 1:-1],  # [b, sample_steps - 1, l, 64]
            "timesteps": timesteps[:, :-1],  # [b, sample_steps - 1]
            "log_probs": torch.cat(all_log_probs)[:, :-1],  # [b, sample_steps - 1]
            "image_id": shared_image_id,  # [l, 3]
            "text_id": data["text_id"],  # [512, 3]
            "prompt_embed_t5": data["prompt_embed_t5"],  # [b, 512, 4096]
            "prompt_embed_clip": data["prompt_embed_clip"],  # [b, 768]
            "sigma_schedule": sigma_schedule,  # [sample_steps + 1]
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
        # Shuffle timesteps and perform training on a random fraction of timesteps 
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
                "unit": "step"
            }

            for t in range(train_timesteps):
                old_log_probs = responses["log_probs"][start_idx:end_idx, t]
                
                # Update model by gradient backward at the end of each group
                should_update_group = (idx + 1) % self.grad_accu_step == 0
                should_step_group = t == train_timesteps - 1
                should_optimizer_step = should_update_group and should_step_group

                def make_forward_fn(s_idx=start_idx, e_idx=end_idx, step=t, p=perms):
                    def forward_fn():
                        prompt_embed_t5 = responses["prompt_embed_t5"][s_idx:e_idx]
                        prompt_embed_clip = responses["prompt_embed_clip"][s_idx:e_idx]
                        latents = responses["latents"][s_idx:e_idx, step]

                        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            model_output = self.model(
                                hidden_states=latents,
                                timestep=responses["timesteps"][s_idx:e_idx, step] / 1000,
                                encoder_hidden_states=prompt_embed_t5,
                                pooled_projections=prompt_embed_clip,
                                txt_ids=responses["text_id"],
                                img_ids=responses["image_id"],
                                guidance=self.sample_cfg,
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
# FLUX Pipeline
# --------------------------------------------------------------------------------
# A simplified version of the diffusers implementation for easy customization.
# NOTE: Some checks have been removed for simplicity, which may increase risk.
# --------------------------------------------------------------------------------
# Usage:
#     from nexus_align.pipelines.pipeline_flux import FluxPipeline
#     (official) from diffusers import FluxPipeline
# --------------------------------------------------------------------------------
class FluxPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    """
    A simplified Flux pipeline for text-to-image generation.

    Reference:
        FLUX: https://blackforestlabs.ai/announcing-black-forest-labs/
        Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
    """

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    )
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> torch.FloatTensor:
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        # Encode prompt
        text_input_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        seq_len = prompt_embeds.shape[1]
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
    ) -> torch.FloatTensor:
        device = device or self._execution_device
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        # Encode prompt
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # Set LoRA scale
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
            # Dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = (
            self.text_encoder.dtype
            if self.text_encoder is not None
            else self.transformer.dtype
        )
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def encode_image(
        self, image: torch.Tensor, device: torch.device, num_images_per_prompt: int
    ) -> torch.Tensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image: PipelineImageInput,
        ip_adapter_image_embeds: list[torch.Tensor],
        device: torch.device,
        num_images_per_prompt: int,
    ) -> list[torch.Tensor]:
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if (
                len(ip_adapter_image)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1
                )
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if (
                len(ip_adapter_image_embeds)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat(
                [single_image_embeds] * num_images_per_prompt, dim=0
            )
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        latent_image_ids = prepare_latent_image_ids(
            height // 2, width // 2, device, dtype
        )
        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids
        else:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = pack_latents(latents, num_channels_latents, height, width)
            return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

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
        prompt: str | list[str] = None,  # prompt for text encoder
        prompt_2: str | list[str] | None = None,  # prompt for text encoder 2
        negative_prompt: str | list[str] = None,  # negative prompt for text encoder
        negative_prompt_2: (
            str | list[str] | None
        ) = None,  # negative prompt for text encoder 2
        true_cfg_scale: float = 1.0,  # classifier-free guidance scale
        height: int | None = None,  # height of the generated image
        width: int | None = None,  # width of the generated image
        num_inference_steps: int = 28,  # number of denoising steps
        sigmas: list[float] | None = None,  # custom sigmas for denoising
        guidance_scale: float = 3.5,  # guidance scale
        num_images_per_prompt: (
            int | None
        ) = 1,  # number of images to generate per prompt
        generator: (
            torch.Generator | list[torch.Generator] | None
        ) = None,  # generator for random noise
        latents: torch.FloatTensor | None = None,  # pre-generated noisy latents
        prompt_embeds: torch.FloatTensor | None = None,  # pre-generated text embeddings
        pooled_prompt_embeds: (
            torch.FloatTensor | None
        ) = None,  # pre-generated pooled text embeddings
        ip_adapter_image: (
            PipelineImageInput | None
        ) = None,  # image input for IP Adapter
        ip_adapter_image_embeds: (
            list[torch.Tensor] | None
        ) = None,  # pre-generated image embeddings for IP Adapter
        negative_ip_adapter_image: (
            PipelineImageInput | None
        ) = None,  # negative image input for IP Adapter
        negative_ip_adapter_image_embeds: (
            list[torch.Tensor] | None
        ) = None,  # pre-generated negative image embeddings for IP Adapter
        negative_prompt_embeds: (
            torch.FloatTensor | None
        ) = None,  # pre-generated negative text embeddings
        negative_pooled_prompt_embeds: (
            torch.FloatTensor | None
        ) = None,  # pre-generated negative pooled text embeddings
        output_type: str | None = "pil",  # output format of the generated image
        return_dict: bool = True,  # whether to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple
        joint_attention_kwargs: dict | None = None,  # kwargs for joint attention
        callback_on_step_end: (
            Callable[[int, int, dict], None] | None
        ) = None,  # callback function on each step end
        callback_on_step_end_tensor_inputs: list[str] = [
            "latents"
        ],  # tensor inputs for callback on step end
        max_sequence_length: int = 512,  # maximum sequence length to use with the prompt
    ) -> FluxPipelineOutput | tuple:
        # --------------------------------------------------------------------------------
        # Prepare
        # --------------------------------------------------------------------------------
        height = (
            height or self.default_sample_size * self.vae_scale_factor
        )  # default: 128 x 8 = 1024
        width = (
            width or self.default_sample_size * self.vae_scale_factor
        )  # default: 128 x 8 = 1024

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # Get batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Get LORA scale
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        # Check if there is a negative prompt
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # --------------------------------------------------------------------------------
        # Encode prompt
        # --------------------------------------------------------------------------------
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = (
                self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )
            )

        # --------------------------------------------------------------------------------
        # Prepare latent
        # --------------------------------------------------------------------------------
        num_channels_latents = self.transformer.config.in_channels // 4  # 64 / 4 = 16
        latents, latent_image_ids = self.prepare_latents(
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
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]

        base_seq_len = self.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = self.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = self.scheduler.config.get("base_shift", 0.5)
        max_shift = self.scheduler.config.get("max_shift", 1.15)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu
        )
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # --------------------------------------------------------------------------------
        # Handle guidance
        # --------------------------------------------------------------------------------
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # --------------------------------------------------------------------------------
        # Denoising loop
        # --------------------------------------------------------------------------------
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                        image_embeds
                    )
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


def prepare_latent_image_ids(
    height: int, width: int, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    """Prepare image position embeddings."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)

    return latent_image_ids.to(device=device, dtype=dtype)


def pack_latents(
    latents: torch.Tensor, num_channels_latents: int, height: int, width: int
) -> torch.Tensor:
    """Pack latent from (B, C, H, W) to (B, H/2, W/2, C*4)."""
    latents = latents.view(-1, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        -1, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
) -> torch.Tensor:
    """Unpack latent from (B, H/2, W/2, C*4) to (B, C, H, W)."""
    batch_size, _, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents
