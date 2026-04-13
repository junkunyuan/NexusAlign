"""FLUX DPO training pipeline."""

import os
from typing import Any

import torch
import torch.distributed as dist

from diffusers.image_processor import VaeImageProcessor

from nexus_align.pipelines.pipeline_flux import (
    FluxTrainPipeline,
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
)
from nexus_align.pipelines.scheduler_flow_match_euler_discrete import (
    RLFlowMatchEulerDiscreteScheduler,
)
from nexus_align.utils.progress import TqdmBar


class FluxDPOTrainPipeline(FluxTrainPipeline):
    """DPO training pipeline: same rollout as GRPO, preference pairs, DPO loss."""

    def __init__(
        self,
        model: Any,
        device: torch.device,
        model_dtype: torch.dtype,
        amp_dtype: torch.dtype,
        kwargs: dict = {},
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            model_dtype=model_dtype,
            amp_dtype=amp_dtype,
            kwargs=kwargs,
        )
        cfg_algo_run = kwargs["algorithm"].get("run", {})
        cfg_model_dpo = kwargs["model"].get("dpo", {})
        self.sample_batch_size = cfg_algo_run["sample_batch_size"]
        self.group_size = cfg_algo_run["group_size"]
        self.sample_save_dir = cfg_algo_run["sample_save_dir"]
        os.makedirs(self.sample_save_dir, exist_ok=True)
        self.sample_height = cfg_model_dpo["sample_height"]
        self.sample_width = cfg_model_dpo["sample_width"]
        self.sample_shift = cfg_model_dpo["sample_shift"]
        self.sample_steps = cfg_model_dpo["sample_steps"]
        self.sample_cfg = torch.tensor(
            [cfg_model_dpo["sample_cfg"]],
            dtype=model_dtype,
            device=device,
        )
        self.sample_eta = cfg_model_dpo["sample_eta"]
        self.train_scheduler_shift = cfg_model_dpo.get(
            "train_scheduler_shift", self.sample_shift
        )
        self.train_sigma_t_mode = cfg_model_dpo.get("train_sigma_t_mode", "uniform")
        self.train_sigma_logit_mean = float(
            cfg_model_dpo.get("train_sigma_logit_mean", 0.0)
        )
        self.train_sigma_logit_std = float(
            cfg_model_dpo.get("train_sigma_logit_std", 1.0)
        )
        self.scheduler = RLFlowMatchEulerDiscreteScheduler(sample_eta=self.sample_eta)
        self.num_train_timesteps_per_pair = cfg_algo_run.get(
            "num_train_timesteps_per_pair", 1
        )

    def _sample_train_flux_sigma(self) -> torch.Tensor:
        """Scalar sigma on device; sigma = s*t/(1+(s-1)*t), t continuous in (0,1)."""
        if self.train_sigma_t_mode == "uniform":
            t = torch.rand((), device=self.device, dtype=self.model_dtype)
        elif self.train_sigma_t_mode == "logit_normal":
            z = torch.randn((), device=self.device, dtype=self.model_dtype)
            z = z * self.train_sigma_logit_std + self.train_sigma_logit_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"❌ unknown train_sigma_t_mode: {self.train_sigma_t_mode}")
        t = t.clamp(min=1e-6, max=1.0 - 1e-6)
        s = self.train_scheduler_shift
        sigma = (s * t) / (1.0 + (s - 1.0) * t)
        return sigma

    @torch.no_grad()
    def sample_responses(self, data: dict) -> dict:
        t = torch.linspace(1, 0, self.sample_steps + 1)
        sigma_schedule = (self.sample_shift * t) / (1 + (self.sample_shift - 1) * t)
        latent_c = 16
        latent_h = self.sample_height // 8
        latent_w = self.sample_width // 8
        shared_latents = torch.randn(
            (1, latent_c, latent_h, latent_w), dtype=self.model_dtype
        )
        shared_latents = pack_latents(shared_latents, latent_c, latent_h, latent_w)
        rank = dist.get_rank()
        batch = len(data["text"])
        batch_ind = torch.arange(batch).chunk(batch // self.sample_batch_size)
        shared_image_id = prepare_latent_image_ids(latent_h // 2, latent_w // 2).to(
            self.device
        )
        all_latents, all_log_probs = list(), list()
        all_clean_latents = list()
        images, image_pils, texts = list(), list(), list()
        bar = TqdmBar(total=len(batch_ind), desc="🚀 DPO sampling", unit="batch")
        self.model.eval()
        for _, b_idx in enumerate(batch_ind):
            batch_size = len(b_idx)
            latents = torch.cat([shared_latents] * batch_size, dim=0).to(self.device)
            prompt_embed_t5 = data["prompt_embed_t5"][b_idx]
            prompt_embed_clip = data["prompt_embed_clip"][b_idx]
            text_id = data["text_id"]
            image_id = shared_image_id
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
            all_clean_latents.append(pred_original.detach())
            texts += [data["text"][t] for t in b_idx]
            self.vae.enable_tiling()
            image_processor = VaeImageProcessor(16)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            ):
                pred_for_vae = unpack_latents(
                    latents=pred_original,
                    height=self.sample_height,
                    width=self.sample_width,
                    vae_scale_factor=8,
                )
                scaling_factor = self.vae.config.scaling_factor
                shift_factor = self.vae.config.shift_factor
                pred_for_vae = pred_for_vae / scaling_factor + shift_factor
                pred_for_vae = pred_for_vae.to(
                    device=self.vae.device, dtype=self.vae.dtype
                )
                image = self.vae.decode(pred_for_vae, return_dict=False)[0]
                img_pil = image_processor.postprocess(image)
            train_state = "-".join(
                [f"{k}{data[k]}" for k in ["epoch", "step", "total_step"]]
            )
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
            "latents": all_latents[:, :-2],
            "next_latents": all_latents[:, 1:-1],
            "clean_latents": torch.cat(all_clean_latents),
            "timesteps": timesteps[:, :-1],
            "log_probs": torch.cat(all_log_probs)[:, :-1],
            "image_id": shared_image_id,
            "text_id": data["text_id"],
            "prompt_embed_t5": data["prompt_embed_t5"],
            "prompt_embed_clip": data["prompt_embed_clip"],
            "sigma_schedule": sigma_schedule,
        }
        torch.cuda.empty_cache()
        return data

    def prepare_data(self, data: dict) -> dict:
        data = super().prepare_data(data)
        if "keys_to_build_groups" not in data:
            data["keys_to_build_groups"] = {
                "text",
                "prompt_embed_t5",
                "prompt_embed_clip",
            }
        return data

    def build_preference_pairs(self, data: dict, meters) -> dict:
        rewards = data["rewards"]
        groups = len(rewards) // self.group_size
        winner_indices = []
        loser_indices = []
        reward_w_list = []
        reward_l_list = []
        for group_idx in range(groups):
            start = group_idx * self.group_size
            end = (group_idx + 1) * self.group_size
            group_rewards = rewards[start:end]
            winner = start + torch.argmax(group_rewards).item()
            loser = start + torch.argmin(group_rewards).item()
            winner_indices.append(winner)
            loser_indices.append(loser)
            reward_w_list.append(rewards[winner])
            reward_l_list.append(rewards[loser])
        reward_w = torch.stack(reward_w_list)
        reward_l = torch.stack(reward_l_list)
        meters.update("reward_margin_mean", (reward_w - reward_l).mean().item())
        data["winner_indices"] = torch.tensor(winner_indices, dtype=torch.long)
        data["loser_indices"] = torch.tensor(loser_indices, dtype=torch.long)
        return data

    def iterate_training_items(self, data: dict):
        winner_indices = data["winner_indices"].to(self.device)
        loser_indices = data["loser_indices"].to(self.device)
        clean_latents = data["clean_latents"].to(self.device)
        num_pairs = len(winner_indices)
        grad_accu_step = self.grad_accu_step
        for pair_idx in range(num_pairs):
            w_idx = winner_indices[pair_idx].item()
            l_idx = loser_indices[pair_idx].item()
            clean_w = clean_latents[w_idx : w_idx + 1]
            clean_l = clean_latents[l_idx : l_idx + 1]
            sigma = self._sample_train_flux_sigma()
            sigma_t = sigma.view(1, 1, 1)
            noise = torch.randn_like(clean_w, dtype=self.model_dtype, device=self.device)
            noisy_w = clean_w + sigma_t * noise
            noisy_l = clean_l + sigma_t * noise
            target_w = noise
            target_l = noise
            timesteps = (sigma * 1000.0).view(1).to(
                device=self.device, dtype=self.model_dtype
            )
            backward_scale = 1.0 / (grad_accu_step * max(1, self.num_train_timesteps_per_pair))
            should_optimizer_step = (pair_idx + 1) % grad_accu_step == 0
            yield {
                "noisy_latents_w": noisy_w,
                "noisy_latents_l": noisy_l,
                "target_w": target_w,
                "target_l": target_l,
                "timesteps": timesteps,
                "prompt_embed_t5": data["prompt_embed_t5"][w_idx : w_idx + 1],
                "prompt_embed_clip": data["prompt_embed_clip"][w_idx : w_idx + 1],
                "text_id": data["text_id"],
                "image_id": data["image_id"],
                "backward_scale": backward_scale,
                "should_optimizer_step": should_optimizer_step,
            }

    def run_trainable_on_pair(self, model, item: dict):
        timesteps = item["timesteps"].to(device=self.device, dtype=self.model_dtype) / 1000
        hidden_states = torch.cat([item["noisy_latents_w"], item["noisy_latents_l"]], dim=0)
        prompt_embed_t5 = torch.cat([item["prompt_embed_t5"], item["prompt_embed_t5"]], dim=0)
        prompt_embed_clip = torch.cat(
            [item["prompt_embed_clip"], item["prompt_embed_clip"]], dim=0
        )
        timesteps = torch.cat([timesteps, timesteps], dim=0)
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            pred_pair = model(
                hidden_states=hidden_states,
                timestep=timesteps,
                encoder_hidden_states=prompt_embed_t5,
                pooled_projections=prompt_embed_clip,
                txt_ids=item["text_id"],
                img_ids=item["image_id"],
                guidance=self.sample_cfg,
            ).sample.to(dtype=self.model_dtype)
        pred_w, pred_l = pred_pair.chunk(2, dim=0)
        return pred_w, pred_l

    def run_reference_on_pair(self, ref_model, item: dict):
        timesteps = item["timesteps"].to(device=self.device, dtype=self.model_dtype) / 1000
        hidden_states = torch.cat([item["noisy_latents_w"], item["noisy_latents_l"]], dim=0)
        prompt_embed_t5 = torch.cat([item["prompt_embed_t5"], item["prompt_embed_t5"]], dim=0)
        prompt_embed_clip = torch.cat(
            [item["prompt_embed_clip"], item["prompt_embed_clip"]], dim=0
        )
        timesteps = torch.cat([timesteps, timesteps], dim=0)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                pred_pair = ref_model(
                    hidden_states=hidden_states,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embed_t5,
                    pooled_projections=prompt_embed_clip,
                    txt_ids=item["text_id"],
                    img_ids=item["image_id"],
                    guidance=self.sample_cfg,
                ).sample.to(dtype=self.model_dtype)
            pred_w, pred_l = pred_pair.chunk(2, dim=0)
        return pred_w, pred_l
