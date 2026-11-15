"""
FDFO algorithm: pull the velocities of two nearby trajectories toward the endpoint with
the higher reward.

For each prompt, FDFO samples a pair of trajectories from one noise, with a small churn
that makes their details differ. The reward difference dR and the endpoint difference dx
of the pair give every velocity along both trajectories the target v - dx / |dx|^2, and
an SPO surrogate loss of a proxy ratio weights it with dR.

License: this file adapts the official code, Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES, under the NVIDIA Source Code License v1 (Non-Commercial). The license is in
reproduce/fdfo/LICENSE.txt, and it limits this file to non-commercial use.

References:
    - FDFO:
        - Paper: Finite Difference Flow Optimization for RL Post-Training of
          Text-to-Image Models (arXiv 2026)
        - Official code: https://github.com/NVlabs/finite-difference-flow-optimization
            - fdfo/sampling.py: edm_step, sample_image_pairs
            - fdfo/train_loop.py: predict_v, and the pair rewards, the target velocity,
              the SPO loss, and the KL to the base model of train_fdfo
            - fdfo/noise_schedules.py: RawChurnScheduler
"""

import numpy as np
import torch

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def edm_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    next_sigma: float,
    churn: float,
) -> torch.Tensor:
    """
    Do one Euler step from sigma to next_sigma with churn; see paper Alg. 1.

    The step overshoots to r < next_sigma, then adds fresh noise that brings the noise
    level back to next_sigma. A churn of 0 gives a plain Euler step.
    """
    gamma = torch.tensor(churn)  # fp32 coefficients, as in the official code
    r = next_sigma / (-gamma * next_sigma + gamma + 1)
    overshoot = latents + (r - sigma) * velocity
    noise_scale = r * ((gamma + 1) ** 2 - 1) ** 0.5 / (gamma * r + 1)
    scale = 1 / (gamma * r + 1)
    return scale * overshoot + noise_scale * torch.randn_like(overshoot)


class FDFO(BaseAlgorithm):
    """FDFO training algorithm."""

    group_size = 2  # images per prompt: one pair of trajectories
    num_micro_steps = 1  # forward passes per training batch of (sample, timestep) slots

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 40)  # rollout sampling steps of the Euler solver
        self.churn_weight = cfg_algorithm.get("churn_weight", 0.0025)  # churn gamma of the initial noise and of each step but the last
        self.guidance_scale = cfg_algorithm.get("guidance_scale", 1.0)  # CFG scale of rollouts and training; 1.0 disables CFG
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Policy loss configs.
        self.clip_range = cfg_algorithm.get("clip_range", 3.0e-2)  # clip range of the SPO ratio, epsilon
        self.kl_weight = cfg_algorithm.get("kl_weight", 0.0)  # coefficient of the velocity KL to the base model

        self.pipeline = None
        self.scheduler = None
        self.neg_embeds = None  # empty-prompt embeddings for CFG

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler

    def predict(self, latents, timesteps, embeds, pooled_embeds, guidance_scale):
        """
        Predict the velocity under autocast.

        With CFG, embeds holds the negative prompts first, as in the diffusers pipeline.
        """
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents, timesteps = torch.cat([latents] * 2), torch.cat([timesteps] * 2)
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            velocity = self.model(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        if do_cfg:
            uncond, cond = velocity.chunk(2)
            velocity = uncond + guidance_scale * (cond - uncond)
        return velocity

    @torch.no_grad()
    def negative_first(self, embeds, pooled_embeds):
        """Put the embeddings of the negative prompt "" first, as in diffusers."""
        if self.neg_embeds is None:
            max_length = self.max_sequence_length
            self.neg_embeds = self.pipeline.encode_prompt([""], max_length)
        neg_embeds, neg_pooled_embeds = self.neg_embeds
        embeds = torch.cat([neg_embeds.expand_as(embeds), embeds])
        pooled_embeds = torch.cat(
            [neg_pooled_embeds.expand_as(pooled_embeds), pooled_embeds]
        )
        return embeds, pooled_embeds

    def forward(self, data):
        """
        Pull the velocity of each (sample, timestep) slot toward its target, with the
        SPO loss; see paper Alg. 2. data["step"] holds the timestep index of each slot.
        """
        rows = torch.arange(len(data["step"]), device=self.device)
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        if self.guidance_scale > 1.0:
            embeds, pooled_embeds = self.negative_first(embeds, pooled_embeds)
        args = (data["xs"][rows, data["step"]], data["timesteps"][rows, data["step"]])
        args += (embeds, pooled_embeds, self.guidance_scale)
        velocity = self.predict(*args)

        # Move the rollout velocity by dx over its squared RMS norm.
        dx = data["dx"]
        dims = tuple(range(1, dx.ndim))
        norm = torch.sqrt(dx.square().mean(dims, keepdim=True)) ** 2
        dx = dx / (norm + 1e-6)
        target = data["velocities"][rows, data["step"]] - dx

        # The proxy ratio is 1 at the rollout velocity.
        log_ratio = -(velocity - target).square().mean(dims) + dx.square().mean(dims)
        ratio = torch.exp(log_ratio)
        dr = data["advantages"]
        penalty = dr.abs() / (2 * self.clip_range) * (ratio - 1) ** 2
        policy_loss = -(dr * ratio) + penalty
        loss = policy_loss.mean()
        metrics = {
            "policy_loss": loss.detach(),
            "clipfrac": ((ratio - 1).abs() > self.clip_range).float().mean().detach(),
        }
        if self.kl_weight > 0:
            with torch.no_grad(), self.model.disable_adapter():
                base_velocity = self.predict(*args)
            kl_loss = (velocity - base_velocity).square().mean()
            loss = loss + self.kl_weight * kl_loss
            metrics["kl_loss"] = kl_loss.detach()
        metrics["loss"] = loss.detach()
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample from data["latents"] with the Euler solver and churn; see paper Alg. 1.

        The last step has no churn. Return the final latents, and the inputs,
        velocities, and timesteps of the model calls.
        """
        latents = data["latents"]
        num_steps = data.get("num_steps", self.num_steps)
        churn = data.get("churn_weight", self.churn_weight)
        guidance_scale = data.get("guidance_scale", self.guidance_scale)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps, sigmas = self.scheduler.timesteps, self.scheduler.sigmas.tolist()
        xs, velocities = [], []
        for i, t in enumerate(timesteps):
            velocity = self.predict(
                latents,
                t.expand(len(latents)),
                data["prompt_embeds"],
                data["pooled_prompt_embeds"],
                guidance_scale,
            )
            xs.append(latents.float())
            velocities.append(velocity)
            gamma = churn if i < num_steps - 1 else 0.0
            latents = edm_step(
                latents.float(), velocity.float(), sigmas[i], sigmas[i + 1], gamma
            )
        return {
            "latents": latents,
            "xs": torch.stack(xs, dim=1),  # (B, num_steps, C, H, W)
            "velocities": torch.stack(velocities, dim=1),  # (B, num_steps, C, H, W)
            "timesteps": timesteps,  # (num_steps,)
        }

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images under autocast."""
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            return self.pipeline.decode(latents)

    def initial_noise(self, num_images: int, dtype: torch.dtype) -> torch.Tensor:
        """Draw the initial noise in the prompt dtype, as diffusers does."""
        latent_size = self.resolution // 8
        shape = (num_images, self.model.config.in_channels, latent_size, latent_size)
        return torch.randn(shape, device=self.device, dtype=dtype)

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """
        Sample one pair of trajectories from one noise for each pair of prompts in a
        row. The churn makes the two images of a pair differ in their details.
        """
        embeds, pooled_embeds = self.pipeline.encode_prompt(
            prompts[0::2], self.max_sequence_length
        )
        latents = self.initial_noise(len(embeds), embeds.dtype)
        embeds, pooled_embeds, latents = (
            x.repeat_interleave(2, dim=0) for x in (embeds, pooled_embeds, latents)
        )

        # Churn the shared initial noise of each trajectory.
        gamma = torch.tensor(self.churn_weight)  # fp32 coefficients, as in edm_step
        scale = 1 / (gamma + 1)
        noise_scale = ((gamma + 1) ** 2 - 1) ** 0.5 / (gamma + 1)
        noise = torch.randn_like(latents)
        latents = scale * latents.float() + noise_scale * noise.float()

        cfg_embeds = (embeds, pooled_embeds)
        if self.guidance_scale > 1.0:
            cfg_embeds = self.negative_first(embeds, pooled_embeds)
        out = self.sample({
            "latents": latents,
            "prompt_embeds": cfg_embeds[0],
            "pooled_prompt_embeds": cfg_embeds[1],
        })

        # Each image of a pair moves by half of the difference from the other image.
        endpoints = out["latents"]
        others = endpoints.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "timesteps": out["timesteps"].repeat(len(embeds), 1),
            "xs": out["xs"],
            "velocities": out["velocities"],
            "dx": (endpoints - others) / 2,
        }
        return {"samples": samples, "responses": self.decode(endpoints)}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images with the Euler ODE sampler, as FDFO evaluates them."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        latents = self.initial_noise(len(prompts), embeds.dtype)
        if guidance_scale > 1.0:
            embeds, pooled_embeds = self.negative_first(embeds, pooled_embeds)
        out = self.sample({
            "latents": latents,
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "num_steps": num_steps,
            "churn_weight": 0.0,
            "guidance_scale": guidance_scale,
        })
        return self.decode(out["latents"])

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """
        Get dR = (R - R_other) / 2 of each image. The two images of a pair are in a row.
        """
        if prompts[0::2] != prompts[1::2]:
            raise ValueError("❌ The two images of a pair must be in a row")
        half = (rewards[0::2] - rewards[1::2]) / 2
        return np.stack([half, -half], axis=1).reshape(-1)
