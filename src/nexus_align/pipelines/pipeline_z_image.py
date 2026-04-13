"""Z-Image evaluation and training pipelines for image generation."""

import os
import random
from typing import Callable, Any

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, PreTrainedModel

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers import ZImageTransformer2DModel, AutoencoderKL
from diffusers.loaders import FromSingleFileMixin, ZImageLoraLoaderMixin
from diffusers.utils.torch_utils import randn_tensor

from nexus_align.core.base_pipeline import BaseTrainPipeline
from nexus_align.pipelines.scheduler_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    RLFlowMatchEulerDiscreteScheduler,
)
from nexus_align.utils.progress import TqdmBar


class ZImageInferPipeline:
    """Inference pipeline for Z-Image."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        kwargs: dict = {},
    ) -> None:
        self.model_name = "ZImage"

        pipeline_path = os.path.join(
            kwargs["common"]["data_and_model_dir"],
            kwargs["model"]["path"],
        )

        # Load components individually, then assemble into local ZImagePipeline
        print(f"⏳ Loading {self.model_name} transformer from <{pipeline_path}>/transformer")
        transformer = ZImageTransformer2DModel.from_pretrained(
            pipeline_path, subfolder="transformer"
        )

        print(f"⏳ Loading {self.model_name} vae from <{pipeline_path}>/vae")
        vae = AutoencoderKL.from_pretrained(pipeline_path, subfolder="vae")

        print(f"⏳ Loading {self.model_name} text_encoder from <{pipeline_path}>/text_encoder")
        from transformers import AutoModel
        text_encoder = AutoModel.from_pretrained(pipeline_path, subfolder="text_encoder")

        print(f"⏳ Loading {self.model_name} tokenizer from <{pipeline_path}>/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(pipeline_path, subfolder="tokenizer")

        print(f"⏳ Loading {self.model_name} scheduler from <{pipeline_path}>/scheduler")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pipeline_path, subfolder="scheduler"
        )

        self._patch_prepare_sequence(transformer)

        pipe = ZImagePipeline(
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
        self.guidance_scale = kwargs["model"]["eval"]["cfg"]
        self.generator = torch.Generator(device=device).manual_seed(
            kwargs["common"]["seed"]
        )

    @staticmethod
    def _patch_prepare_sequence(transformer) -> None:
        """Replace ``_prepare_sequence`` with a variant that avoids the
        CUDA advanced-indexing kernel bug triggered by
        ``feats_cat[bool_mask] = pad_token``.

        Uses ``torch.where`` for element-wise selection instead.
        """
        from torch.nn.utils.rnn import pad_sequence

        model_cls = type(transformer)

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


class ZImageTrainPipeline(BaseTrainPipeline):
    """Training pipeline for Z-Image."""

    # Z-Image uses AutoencoderKL with standard scaling_factor/shift_factor.
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
            data["keys_to_build_groups"] = {"text", "prompt_embeds"}

        return data

    def _call_transformer(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: list[torch.Tensor],
        model=None,
    ) -> torch.Tensor:
        """
        Call a Z-Image transformer with its list-based interface.

        The Z-Image transformer expects per-sample lists for hidden_states
        and prompt_embeds, with a reversed-and-normalized timestep format.

        Args:
            model: transformer to call. Defaults to self.model (trainable).
        """
        if model is None:
            model = self.model

        # Z-Image timestep: (1000 - t) / 1000 → 0 at full noise, 1 at clean
        z_timestep = (1000.0 - timesteps.float()) / 1000.0

        # Transformer expects (B, C, 1, H, W) input as a list per sample
        latent_5d = latents.unsqueeze(2)
        latent_list = list(latent_5d.unbind(dim=0))

        model_out_list = model(
            latent_list, z_timestep, prompt_embeds, return_dict=False
        )[0]

        # Stack back, remove temporal dim, negate (Z-Image convention)
        model_output = torch.stack(
            [t.to(self.model_dtype) for t in model_out_list], dim=0
        )
        model_output = model_output.squeeze(2)
        model_output = -model_output

        return model_output

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        """Rollout: generate responses from the prepared data."""
        # Prepare denoising schedule
        t = torch.linspace(1, 0, self.sample_steps + 1)
        sigma_schedule = (self.sample_shift * t) / (1 + (self.sample_shift - 1) * t)

        # Prepare init latents (Z-Image uses standard spatial latents, no packing)
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
            prompt_embeds = [data["prompt_embeds"][i] for i in b_idx]

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
                    model_output = self._call_transformer(
                        latents, timesteps, prompt_embeds
                    )

                # Flatten spatial dims for step_fn
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
            image_processor = VaeImageProcessor(self.VAE_SCALE_FACTOR * 2)
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
                file_name = f"z_image-{train_state}-rank{rank}-{img_idx}.png"
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
        )

        data = {
            "reward_inputs": {"image": images, "image_pil": image_pils, "text": texts},
            "latents": latents_trimmed,
            "next_latents": next_latents_trimmed,
            "timesteps": timesteps_trimmed,
            "log_probs": torch.cat(all_log_probs)[:, :-1],
            "ref_log_probs": ref_log_probs,
            "prompt_embeds": data["prompt_embeds"],
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
        prompt_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probs from the frozen reference model for KL divergence."""
        if self.ref_model is None:
            return torch.zeros_like(latents[:, :, 0, 0])

        batch, num_steps = latents.shape[0], latents.shape[1]
        batch_ind = torch.arange(batch).chunk(batch // self.sample_batch_size)

        all_ref_log_probs = []
        bar = TqdmBar(total=len(batch_ind), desc="🔒 Computing ref log probs", unit="batch")
        self.ref_model.eval()

        for b_idx in batch_ind:
            ref_log_probs_steps = []
            step_prompt_embeds = [prompt_embeds[j] for j in b_idx]

            for i in range(num_steps):
                step_latents = latents[b_idx, i].to(self.device)
                step_next_latents = next_latents[b_idx, i].to(self.device)
                step_timesteps = timesteps[b_idx, i].to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=self.amp_dtype
                ):
                    model_output = self._call_transformer(
                        step_latents, step_timesteps, step_prompt_embeds,
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
                        latents = responses["latents"][s_idx:e_idx, step]
                        timesteps = responses["timesteps"][s_idx:e_idx, step]

                        with torch.amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        ):
                            model_output = self._call_transformer(
                                latents, timesteps, prompt_embeds
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
# Z-Image Pipeline
# --------------------------------------------------------------------------------
# A simplified version of the diffusers implementation for easy customization.
# NOTE: Some checks have been removed for simplicity, which may increase risk.
# --------------------------------------------------------------------------------
# Usage:
#     from nexus_align.pipelines.pipeline_z_image import ZImagePipeline
#     (official) from diffusers import ZImagePipeline
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


class ZImagePipeline(
    DiffusionPipeline,
    ZImageLoraLoaderMixin,
    FromSingleFileMixin,
):
    """
    A simplified Z-Image pipeline for text-to-image generation.

    Reference:
        Z-Image: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
        Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        max_sequence_length: int = 512,
    ) -> list[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        # Apply Qwen3 chat template with thinking enabled
        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Use second-to-last hidden state
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        # Extract only valid (non-padding) tokens per prompt → variable-length list
        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[list[torch.FloatTensor], list[torch.FloatTensor]]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = (
                    [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []

        return prompt_embeds, negative_prompt_embeds

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
        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

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
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ) -> ZImagePipelineOutput | tuple:
        # --------------------------------------------------------------------------------
        # Prepare
        # --------------------------------------------------------------------------------
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # --------------------------------------------------------------------------------
        # Encode prompt
        # --------------------------------------------------------------------------------
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # --------------------------------------------------------------------------------
        # Prepare latents
        # --------------------------------------------------------------------------------
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [
                pe for pe in prompt_embeds for _ in range(num_images_per_prompt)
            ]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [
                    npe
                    for npe in negative_prompt_embeds
                    for _ in range(num_images_per_prompt)
                ]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # --------------------------------------------------------------------------------
        # Prepare timesteps
        # --------------------------------------------------------------------------------
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
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

        # --------------------------------------------------------------------------------
        # Denoising loop
        # --------------------------------------------------------------------------------
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0])
                # Z-Image uses reversed timestep: (1000 - t) / 1000
                timestep = (1000 - timestep) / 1000
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                apply_cfg = (
                    self.do_classifier_free_guidance and current_guidance_scale > 0
                )

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                # Z-Image transformer expects list inputs with temporal dim
                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    return_dict=False,
                )[0]

                if apply_cfg:
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        if (
                            self._cfg_normalization
                            and float(self._cfg_normalization) > 0.0
                        ):
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(
                                self._cfg_normalization
                            )
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack(
                        [t.float() for t in model_out_list], dim=0
                    )

                noise_pred = noise_pred.squeeze(2)
                # Z-Image convention: negate noise prediction
                noise_pred = -noise_pred

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

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
            latents = latents.to(self.vae.dtype)
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
