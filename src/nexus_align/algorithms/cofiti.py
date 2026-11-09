# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Portions adapted from the local finite-difference-flow-optimization source.
# Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.
"""CoFiTi: group preference projection and endpoint-copy velocity MSE.

The fixed nonlinear projection and churn sampling are local FDFO extensions.
Source: fdfo/exact_projection.py, fdfo/sampling.py, and fdfo/sa_solver.py.
Background: https://arxiv.org/abs/2603.12893 and
https://github.com/NVlabs/finite-difference-flow-optimization.
"""

import math

import torch

from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.algorithms.components.cofiti_sampling import sample_euler, sample_sa
from nexus_align.algorithms.components.cofiti_projection import (
    solve_fixed_exact_projection,
)


def require_finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


class CoFiTi(BaseAlgorithm):
    """One group projection, one frozen endpoint-copy target, all-sample MSE."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)
        cfg = cfg_algorithm

        # Sampling configs.
        self.solver = cfg.get("solver", 'euler')  # rollout solver: euler or sa
        self.optimization_level = cfg.get(
            "optimization_level", 'trajectory'
        )  # trajectory or distribution initial noise
        self.resolution = cfg.get("resolution", 512)  # image resolution
        self.num_steps = cfg.get("num_steps", 10)  # model evaluations; sa requires >= 2
        self.guidance_scale = cfg.get("guidance_scale", 1.0)  # classifier-free guidance
        self.churn_weight = cfg.get("churn_weight", 0.04)  # rollout noise strength
        self.max_sequence_length = cfg.get("max_sequence_length", 128)  # T5 tokens
        self.max_batch_size = cfg.get("max_batch_size", 28)  # forward batch ceiling
        self.vae_decode_chunk_size = cfg.get("vae_decode_chunk_size", 16)  # VAE batch
        self.deduplicate_prompt_embeddings = cfg.get(
            "deduplicate_prompt_embeddings", True
        )  # share embeddings within each prompt draw

        # Group projection configs.
        self.group_size = cfg.get("group_size", 24)  # members per prompt draw
        self.eta = cfg.get("eta", 1.0)  # preference tilt; null means 1/group size
        self.u_scale = cfg.get("u_scale", 0.1)  # multiply the solved FP64 correction
        self.exact_projection_iterations = cfg.get(
            "exact_projection_iterations", 5
        )  # fixed Newton update count
        self.ridge_scale = cfg.get("ridge_scale", 0.001)  # mean Gram diagonal factor
        self.ridge_eps = cfg.get(
            "ridge_eps", 1e-06
        )  # positive projection regularization floor

        # Training regularization configs.
        self.kl_weight = cfg.get(
            "kl_weight", 0.004
        )  # adapter-disabled velocity regularization

        allowed = {
            'churn_weight',
            'deduplicate_prompt_embeddings',
            'eta',
            'exact_projection_iterations',
            'group_size',
            'guidance_scale',
            'kl_weight',
            'max_batch_size',
            'max_sequence_length',
            'name',
            'num_steps',
            'optimization_level',
            'resolution',
            'ridge_eps',
            'ridge_scale',
            'solver',
            'u_scale',
            'vae_decode_chunk_size',
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported CoFiTi configuration fields: {sorted(unknown)}"
            )
        if self.eta is None:
            self.eta = 1.0 / self.group_size
        self.wrapper = None
        self.scheduler = None
        self.validate_config()

    def validate_config(self):
        if self.solver not in {"euler", "sa"}:
            raise ValueError("CoFiTi solver must be euler or sa")
        if self.optimization_level not in {"trajectory", "distribution"}:
            raise ValueError("optimization_level must be trajectory or distribution")
        for name in ("group_size", "num_steps", "exact_projection_iterations",
                     "max_batch_size", "vae_decode_chunk_size", "max_sequence_length"):
            value = getattr(self, name)
            minimum = 2 if name == "group_size" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.solver == "sa" and self.num_steps < 2:
            raise ValueError("CoFiTi solver=sa requires num_steps >= 2")
        if self.resolution < 8 or self.resolution % 8:
            raise ValueError("resolution must be a positive multiple of eight")
        for name in ("ridge_scale", "kl_weight", "churn_weight", "guidance_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("ridge_eps", "eta", "u_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def prepare(self, model, device, pipeline=None):
        """Bind the SD3 wrapper; its model is DDP, transformer is the raw PEFT model."""
        self.wrapper = model if pipeline is None else pipeline
        student = model.model if pipeline is None else model
        super().prepare(student, torch.device(device))
        self.scheduler = self.wrapper.scheduler

    def autocast(self):
        """Use the source SD3.5 training precision for rollout and optimization."""
        return torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        )

    @torch.no_grad()
    def build_targets(self, final_latents, rewards):
        """Solve and scale one correction per group, without activity masking."""
        count, k = final_latents.shape[0], self.group_size
        if count == 0 or count % k or rewards.numel() != count:
            raise ValueError("endpoints and rewards must contain complete groups")
        endpoints = final_latents.reshape(count // k, k, *final_latents.shape[1:])
        grouped_rewards = rewards.reshape(count // k, k)
        u = solve_fixed_exact_projection(
            endpoints, grouped_rewards, self.eta,
            iterations=self.exact_projection_iterations,
            ridge_scale=self.ridge_scale, ridge_eps=self.ridge_eps,
        )
        displacement = (u * self.u_scale).float().detach()
        require_finite("projection displacement", displacement)
        return {
            "displacement": displacement.repeat_interleave(k, dim=0),
            "metrics": {
                "group_reward_std": (
                    grouped_rewards.float().std(1, unbiased=False).mean()
                ),
                "projection_rms": displacement.double().square().mean().sqrt(),
            },
        }

    def loss_from_predictions(self, prediction, reference, base, data):
        """Every member/time sample contributes, including zero displacement."""
        displacement = data["displacement"].detach().float()
        if (displacement.shape != prediction.shape
                or reference.shape != prediction.shape):
            raise ValueError("velocity and displacement shapes must match")
        target = (reference.float() - displacement).detach()
        dims = tuple(range(1, prediction.ndim))
        policy_loss = (prediction.float() - target).square().mean(dims).mean()
        kl_loss = prediction.new_zeros((), dtype=torch.float32)
        if self.kl_weight:
            if base is None or base.shape != prediction.shape:
                raise ValueError("base velocity is required for regularization")
            # Keep source autocast subtraction/square precision for the base term.
            kl_loss = (prediction - base.detach()).square().mean(dims).mean()
        loss = policy_loss + self.kl_weight * kl_loss
        return {
            "loss": loss,
            "metrics": {
                "loss": loss.detach(), "policy_loss": policy_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "target_delta_rms": displacement.square().mean().sqrt().detach(),
            },
        }

    def forward(self, data):
        do_cfg = abs(self.guidance_scale - 1.0) > 1e-6
        hidden, times = data["xs"], data["timesteps"]
        text, pool = data["prompt_embeds"], data["pooled_prompt_embeds"]
        if do_cfg:
            negative = data["negative_prompt_embeds"].expand_as(text)
            negative_pool = data["negative_pooled_prompt_embeds"].expand_as(pool)
            hidden, times = torch.cat([hidden, hidden]), torch.cat([times, times])
            text, pool = torch.cat([negative, text]), torch.cat([negative_pool, pool])
        inputs = dict(
            hidden_states=hidden,
            timestep=times,
            encoder_hidden_states=text,
            pooled_projections=pool,
            return_dict=False,
        )
        with self.autocast():
            prediction = self.model(**inputs)[0]
        base = None
        if self.kl_weight:
            # Separate autocast cache regions prevent stale casts after adapter swaps.
            with torch.no_grad(), self.wrapper.transformer.disable_adapter():
                with self.autocast():
                    base = self.wrapper.transformer(**inputs)[0]
        if do_cfg:
            negative, positive = prediction.chunk(2)
            prediction = negative + self.guidance_scale * (positive - negative)
            if base is not None:
                negative, positive = base.chunk(2)
                base = negative + self.guidance_scale * (positive - negative)
        with self.autocast():
            return self.loss_from_predictions(
                prediction, data["velocities"], base, data
            )

    @torch.no_grad()
    def sample(self, data):
        """Draw complete groups from pre-encoded prompts and record model calls."""
        embeds = data["prompt_embeds"]
        pooled = data["pooled_prompt_embeds"]
        group_size = self.group_size
        count = embeds.shape[0]
        member_indices = torch.arange(count, device=self.device).repeat_interleave(
            group_size
        )
        member_embeds, member_pooled = embeds[member_indices], pooled[member_indices]
        guidance = self.guidance_scale
        do_cfg = abs(guidance - 1.0) > 1e-6
        if do_cfg:
            negative = data["negative_prompt_embeds"]
            negative_pooled = data["negative_pooled_prompt_embeds"]
            if negative.shape[0] == 1:
                negative = negative.expand(count, -1, -1)
                negative_pooled = negative_pooled.expand(count, -1)
            member_negative = negative[member_indices]
            member_negative_pooled = negative_pooled[member_indices]
        shape = (
            count if self.optimization_level == "trajectory" else count * group_size,
            self.wrapper.transformer.config.in_channels,
            self.resolution // 8,
            self.resolution // 8,
        )
        x = torch.randn(shape, device=self.device, dtype=embeds.dtype)
        num_steps = self.num_steps
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        if self.optimization_level == "trajectory":
            x = x.repeat_interleave(group_size, dim=0)
            if self.churn_weight > 0:
                gamma = torch.full((x.shape[0],), self.churn_weight, device=self.device)
                scale = (1 / (gamma + 1))[:, None, None, None]
                noise_scale = (((gamma + 1).square() - 1).sqrt() / (gamma + 1))
                x = scale * x + noise_scale[:, None, None, None] * torch.randn_like(x)

        def predict_velocity(sample, timestep):
            predictions = []
            for start in range(0, sample.shape[0], self.max_batch_size):
                end = start + self.max_batch_size
                hidden = sample[start:end]
                text, pool = member_embeds[start:end], member_pooled[start:end]
                if do_cfg:
                    hidden = torch.cat([hidden, hidden])
                    text = torch.cat([member_negative[start:end], text])
                    pool = torch.cat([member_negative_pooled[start:end], pool])
                with self.autocast():
                    velocity = self.wrapper.transformer(
                        hidden_states=hidden,
                        timestep=timestep.expand(hidden.shape[0]),
                        encoder_hidden_states=text,
                        pooled_projections=pool,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    negative_v, positive_v = velocity.chunk(2)
                    velocity = negative_v + guidance * (positive_v - negative_v)
                predictions.append(velocity)
            return torch.cat(predictions)

        sampler = sample_euler if self.solver == "euler" else sample_sa
        output = sampler(
            predict_velocity,
            x,
            timesteps,
            self.scheduler.sigmas[:num_steps + 1],
            self.churn_weight,
        )
        with self.autocast():
            images = torch.cat([
                self.wrapper.decode(chunk)
                for chunk in output.sample.split(self.vae_decode_chunk_size)
            ])
        if not self.deduplicate_prompt_embeddings:
            embeds, pooled = member_embeds, member_pooled
            member_indices = torch.arange(x.shape[0], device=self.device)
        return {
            "images": images,
            "final_latents": output.sample,
            "xs": torch.stack(output.model_inputs, dim=1),
            "velocities": torch.stack(output.velocities, dim=1),
            "timesteps": output.model_timesteps,
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled,
            "prompt_indices": member_indices,
        }

    @torch.no_grad()
    def generate(
        self, prompts, num_steps, guidance_scale, *, seeds=None,
        resolution=512, max_sequence_length=256,
    ):
        """Evaluate the student with the source's native Diffusers Flow Euler path."""
        return self.wrapper.generate_evaluation(
            prompts, num_steps, guidance_scale,
            seeds=[0] * len(prompts) if seeds is None else seeds,
            resolution=resolution, max_sequence_length=max_sequence_length,
        )
