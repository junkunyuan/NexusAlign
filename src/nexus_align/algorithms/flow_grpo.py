"""
Flow-GRPO algorithm: SDE sampling with log probabilities, and the clipped policy loss.

References:
    - Flow-GRPO:
        - Paper: Flow-GRPO: Training Flow Matching Models via Online RL (NeurIPS 2025)
        - Official code: https://github.com/yifan123/flow_grpo
            - flow_grpo/diffusers_patch/sd3_sde_with_logprob.py: sde_step_with_logprob
            - flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:
              pipeline_with_logprob
            - scripts/train_sd3.py: compute_log_prob, and the clipped policy loss and
              KL of main
            - flow_grpo/stat_tracking.py: PerPromptStatTracker
"""

import math
import random

import numpy as np
import torch

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def sde_step_with_logprob(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    sigma_max: float,
    noise_level: float,
    prev_sample: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Do one reverse SDE step from sigma to sigma_prev; see paper Eq. 9.

    If prev_sample is None, sample it. Return its Gaussian log probability, as the
    mean over all dimensions except the batch dimension. sigma_max replaces sigma = 1
    in the noise std.
    """
    # prev_sample_mean can overflow in bf16, so calculate it in fp32.
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    dt = sigma_prev - sigma
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
    std_dev_t = std_dev_t * noise_level
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    std = std_dev_t * torch.sqrt(-1 * dt)  # std of prev_sample
    if prev_sample is None:
        noise = torch.randn(
            model_output.shape, device=model_output.device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std * noise
    
    # Log probability: log N(x;μ,σ2) = −(x−μ)**2/2σ**2 ​−logσ −1/2​*log(2π)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std**2)
        - torch.log(std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def group_advantages(prompts: list[str], rewards: np.ndarray) -> np.ndarray:
    """Subtract each prompt group's mean and divide by the std of all rewards."""
    prompts = np.array(prompts)
    rewards = np.asarray(rewards, dtype=np.float64)
    advantages = np.zeros_like(rewards)
    std = np.std(rewards, axis=0, keepdims=True) + 1e-4
    for prompt in np.unique(prompts):
        group = rewards[prompts == prompt]
        mean = np.mean(group, axis=0, keepdims=True)
        advantages[prompts == prompt] = (group - mean) / std
    return advantages


class FlowGRPO(BaseAlgorithm):
    """Flow-GRPO training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 10)  # denoising steps to collect samples (denoising reduction)
        self.guidance_scale = cfg_algorithm.get("guidance_scale", 4.5)  # CFG scale for sampling and training; 1.0 disables CFG
        self.noise_level = cfg_algorithm.get("noise_level", 0.7)  # SDE noise level a; see paper Eq. 9
        self.sde_window_size = cfg_algorithm.get("sde_window_size", 0)  # if > 0, use the SDE and train only in a random window of this many steps (Flow-GRPO-Fast)
        self.sde_window_range = cfg_algorithm.get("sde_window_range", [0, 5])  # the window lies in [low, high)
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Advantage configs.
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt, G

        # Policy loss configs.
        self.timestep_fraction = cfg_algorithm.get("timestep_fraction", 0.99)  # train on the first int(num_steps * fraction) steps
        self.clip_range = cfg_algorithm.get("clip_range", 1.0e-4)  # clip range of the PPO ratio, epsilon
        self.adv_clip_max = cfg_algorithm.get("adv_clip_max", 5.0)  # clip advantages to [-max, max]
        self.beta = cfg_algorithm.get("beta", 0.04)  # KL coefficient; see paper Eq. 5

        self.pipeline = None
        self.scheduler = None
        self.sigmas = None  # schedule of the latest rollout
        self.sigma_max = None
        self.neg_prompt_embeds = None
        self.neg_pooled_prompt_embeds = None
        # Forward passes per training batch, one per trained denoising step.
        self.num_micro_steps = int(self.num_steps * self.timestep_fraction)
        if self.sde_window_size > 0:
            self.num_micro_steps = self.sde_window_size

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        # The empty negative prompt of CFG.
        self.neg_prompt_embeds, self.neg_pooled_prompt_embeds = pipeline.encode_prompt(
            [""], self.max_sequence_length
        )

    def predict(self, latents, timesteps, embeds, pooled_embeds, guidance_scale):
        """
        Predict the velocity, with CFG if guidance_scale > 1.0.

        For CFG, embeds must stack the [negative, positive] prompts.
        """
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents = torch.cat([latents] * 2)
            timesteps = torch.cat([timesteps] * 2)
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            noise_pred = self.model(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        noise_pred = noise_pred.float()
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = (
                noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            )
        return noise_pred

    def forward(self, data):
        """Calculate the GRPO loss at the stored step data["step"]; see paper Eq. 5."""
        # 1. Get the stored step and prompts, with the negative prompts first for CFG.
        j = data["step"]
        latents, next_latents = data["latents"][:, j], data["next_latents"][:, j]
        old_log_prob, timesteps = data["log_probs"][:, j], data["timesteps"][:, j]
        steps = data["steps"][:, j].view(-1, 1, 1, 1)
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        if self.guidance_scale > 1.0:
            batch_size = embeds.shape[0]
            neg_embeds = self.neg_prompt_embeds.repeat(batch_size, 1, 1)
            neg_pooled_embeds = self.neg_pooled_prompt_embeds.repeat(batch_size, 1)
            embeds = torch.cat([neg_embeds, embeds])
            pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])
        sigmas = self.sigmas
        sigma, sigma_prev, sigma_max = sigmas[steps], sigmas[steps + 1], self.sigma_max

        # 2. Get the log probability of the stored next latent under the current policy.
        noise_pred = self.predict(latents, timesteps, embeds, pooled_embeds, self.guidance_scale)
        _, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            noise_pred, latents, sigma, sigma_prev, sigma_max, self.noise_level,
            prev_sample=next_latents,
        )

        # 3. Calculate the clipped policy loss.
        advantages = data["advantages"]
        advantages = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = torch.mean(torch.maximum(-advantages * ratio, -advantages * clipped_ratio))
        metrics = {"policy_loss": policy_loss.detach()}

        # 4. Calculate the KL to the reference policy (LoRA disabled) in closed form;
        # see paper Eq. 11.
        loss = policy_loss
        if self.beta > 0:
            with torch.no_grad(), self.model.disable_adapter():
                noise_pred_ref = self.predict(
                    latents, timesteps, embeds, pooled_embeds, self.guidance_scale
                )
                _, _, prev_sample_mean_ref, _ = sde_step_with_logprob(
                    noise_pred_ref, latents, sigma, sigma_prev, sigma_max, self.noise_level,
                    prev_sample=next_latents,
                )
            kl_loss = (prev_sample_mean - prev_sample_mean_ref) ** 2
            kl_loss = kl_loss.mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t**2)
            kl_loss = torch.mean(kl_loss)
            loss = policy_loss + self.beta * kl_loss
            metrics["kl_loss"] = kl_loss.detach()

        metrics.update({
            "loss": loss.detach(),
            "approx_kl": 0.5 * torch.mean((log_prob - old_log_prob) ** 2).detach(),
            "clipfrac": torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()),
            "clipfrac_gt_one": torch.mean((ratio - 1.0 > self.clip_range).float()),
            "clipfrac_lt_one": torch.mean((1.0 - ratio > self.clip_range).float()),
        })
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample latents with the SDE sampler. Keep each step and its log probability.

        noise_level = 0 gives the deterministic ODE sampler for evaluation.
        """
        # 1. Prepare the prompts, with the negative prompts first for CFG.
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        guidance_scale = data.get("guidance_scale", self.guidance_scale)
        noise_level = data.get("noise_level", self.noise_level)
        start, end = data.get("sde_window", (0, None))
        num_steps = data.get("num_steps", self.num_steps)
        batch_size = embeds.shape[0]
        if guidance_scale > 1.0:
            embeds = torch.cat([data["neg_prompt_embeds"][:batch_size], embeds])
            pooled_embeds = torch.cat(
                [data["neg_pooled_prompt_embeds"][:batch_size], pooled_embeds]
            )

        # 2. Draw the initial noise in the prompt dtype, as the diffusers pipeline does.
        latent_size = self.resolution // 8
        latents = torch.randn(
            (batch_size, self.model.config.in_channels, latent_size, latent_size),
            device=self.device, dtype=embeds.dtype,
        ).float()

        # 3. Denoise with the SDE, and record each step.
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps.clone()
        sigmas = self.scheduler.sigmas.clone()
        sigma_max = sigmas[1].item()
        all_latents, all_log_probs = [latents], []
        for i, t in enumerate(timesteps):
            noise_pred = self.predict(
                latents, t.expand(batch_size), embeds, pooled_embeds, guidance_scale
            )
            # Use ODE steps (noise level 0) outside the SDE window.
            in_window = start <= i and (end is None or i < end)
            latents, log_prob, _, _ = sde_step_with_logprob(
                noise_pred, latents, sigmas[i], sigmas[i + 1], sigma_max,
                noise_level if in_window else 0.0,
            )
            all_latents.append(latents)
            all_log_probs.append(log_prob)

        return {
            "latents": torch.stack(all_latents, dim=1),  # (B, num_steps + 1, C, H, W)
            "log_probs": torch.stack(all_log_probs, dim=1),  # (B, num_steps)
            "timesteps": timesteps.repeat(batch_size, 1),  # (B, num_steps)
            "sigmas": sigmas,  # (num_steps + 1,)
        }

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images in fp32."""
        return self.pipeline.decode(latents)

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """Sample one image per prompt. Return the images and their training data."""
        # Prepare the positive and negative prompt embeddings.
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        batch_size = len(prompts)
        neg_embeds = self.neg_prompt_embeds.repeat(batch_size, 1, 1)
        neg_pooled_embeds = self.neg_pooled_prompt_embeds.repeat(batch_size, 1)
        
        # Choose the SDE window.
        start, end = 0, self.num_micro_steps
        if self.sde_window_size > 0:
            low, high = self.sde_window_range
            start = random.randint(low, high - self.sde_window_size)
            end = start + self.sde_window_size
        
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "neg_prompt_embeds": neg_embeds,
            "neg_pooled_prompt_embeds": neg_pooled_embeds,
            "sde_window": (start, end) if self.sde_window_size > 0 else (0, None),
        })
        self.sigmas = out["sigmas"]
        self.sigma_max = self.sigmas[1].item()  # read once, because reads sync the GPU
        latents = out["latents"]
        # T = end - start trained steps.
        steps = torch.arange(start, end, device=self.device)
        samples = {
            "prompt_embeds": embeds,  # (sample_batch_size, 77 + max_sequence_length, 4096)
            "pooled_prompt_embeds": pooled_embeds,  # (sample_batch_size, 2048)
            "steps": steps.repeat(batch_size, 1),  # (sample_batch_size, T)
            "timesteps": out["timesteps"][:, start:end],  # (sample_batch_size, T)
            "latents": latents[:, start:end],  # (sample_batch_size, T, C, H, W)
            "next_latents": latents[:, start + 1:end + 1],  # (sample_batch_size, T, C, H, W)
            "log_probs": out["log_probs"][:, start:end],  # (sample_batch_size, T)
        }
        images = self.decode(out["latents"][:, -1])
        return {"samples": samples, "responses": images}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images at the given CFG scale, with noise level 0 (ODE)."""
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, self.max_sequence_length)
        batch_size = len(prompts)
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "neg_prompt_embeds": self.neg_prompt_embeds.repeat(batch_size, 1, 1),
            "neg_pooled_prompt_embeds": self.neg_pooled_prompt_embeds.repeat(batch_size, 1),
            "guidance_scale": guidance_scale,
            "noise_level": 0.0,
            "num_steps": num_steps,
        })
        return self.decode(out["latents"][:, -1])

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        return group_advantages(prompts, rewards)
