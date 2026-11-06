"""
DiffusionART algorithm: regress rollout velocities to a shared reward guidance.

References:
    - DiffusionART:
        - Paper: DiffusionART: Efficient, Stable and Scalable Reinforcement of
          Diffusion Models (arXiv 2026)
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/algorithms/covariance.py: art_scalar_covariance, for the
              guidance of each prompt group
            - src/visual_rl/algorithms/art.py: art_loss, for the frozen target, the
              sigma weight, and the trust-region mask
            - src/visual_rl/training/rollout.py: collect_rollout, for the Euler ODE
              rollouts and their cached velocities
            - src/visual_rl/backends/sd3.py: evaluate, for the evaluation sampler
"""

import hashlib

import numpy as np
import torch
import torch.distributed as dist

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.engine.distributed import gather_tensor
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def prompt_ids(prompts: list[str]) -> torch.Tensor:
    """Hash each prompt to an int64 that is the same on all ranks."""
    return torch.tensor([
        int.from_bytes(hashlib.sha256(p.encode()).digest()[:8], "big", signed=True)
        for p in prompts
    ])


class DiffusionART(BaseAlgorithm):
    """DiffusionART training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 10)  # rollout sampling steps of the Euler ODE
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Guidance configs.
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt, K

        # Trust-region mask configs.
        self.mask_noise_level = cfg_algorithm.get("mask_noise_level", 0.8)  # noise level nu of the reference SDE; see paper Appendix C.2
        self.mask_threshold = cfg_algorithm.get("mask_threshold", 1.0e-5)  # drift threshold delta; see paper Eq. 8

        self.pipeline = None
        self.scheduler = None
        self.sigmas = None  # noise levels of the rollout steps
        # Forward passes per training batch, one per sampling timestep.
        self.num_micro_steps = self.num_steps

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        self.scheduler.set_timesteps(self.num_steps, device=device)
        self.sigmas = self.scheduler.sigmas.float().clone()

    def predict(self, latents, sigma, embeds, pooled_embeds) -> torch.Tensor:
        """Predict the fp32 velocity at the float timestep sigma * 1000."""
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            velocity = self.model(
                hidden_states=latents,
                timestep=(sigma * 1000).expand(latents.shape[0]),
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        return velocity.float()

    def forward(self, data):
        """Calculate the ART loss at the data["step"]-th timestep; see paper Eq. 6."""
        # 1. Predict the velocity at the cached rollout state of this timestep.
        j = data["step"]
        xt, old = data["latents"][:, j], data["velocities"][:, j]
        sigma, dt = self.sigmas[j], self.sigmas[j + 1] - self.sigmas[j]
        pred = self.predict(
            xt, sigma, data["prompt_embeds"], data["pooled_prompt_embeds"]
        )

        # 2. Regress it to the frozen target. The velocity points from the clean
        # latents to the noise, so the target subtracts the guidance.
        guidance = data["guidance"]
        target = old - guidance
        dims = tuple(range(1, xt.ndim))
        mse = ((pred - target) ** 2).mean(dim=dims)  # see paper Eq. 6

        # 3. Mask the outward moves whose drift from the old velocity is too large.
        # The drift is the KL of the reference SDE steps with the current and the old
        # velocities, per latent dimension; see paper Eq. 7 and 8.
        with torch.no_grad():
            drift = pred - old
            nu = self.mask_noise_level
            if sigma == 1:  # the first step uses the endpoint convention
                coef, var = dt, nu**2
            else:
                coef, var = dt * (1 + nu**2 / 2), nu**2 * sigma / (1 - sigma) * -dt
            kl = coef**2 * drift.square().mean(dim=dims) / (2 * var)
            outward = (drift * (target - pred)).mean(dim=dims) > 0
            masked = (kl >= self.mask_threshold) & outward
            # A group with equal rewards has zero guidance and no loss.
            active = guidance.flatten(1).abs().amax(dim=1) > 0
            keep = active & ~masked

        # 4. Weight each loss by sigma, and average over all samples.
        loss = (sigma * mse * keep).mean()
        metrics = {
            "masked_ratio": masked.float().mean(),
            "active_ratio": active.float().mean(),
            "drift_kl": kl.mean(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample fp32 latents with the Euler ODE, as the official collect_rollout does.

        Return the states before and after each step, and the velocity at each state.
        """
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        num_steps = data.get("num_steps", self.num_steps)
        latent_size = self.resolution // 8
        latents = torch.randn(
            (embeds.shape[0], self.model.config.in_channels, latent_size, latent_size),
            device=self.device,
        )
        self.scheduler.set_timesteps(num_steps, device=self.device)
        sigmas = self.scheduler.sigmas.float()
        all_latents, velocities = [latents], []
        for i in range(num_steps):
            velocity = self.predict(latents, sigmas[i], embeds, pooled_embeds)
            latents = latents + velocity * (sigmas[i + 1] - sigmas[i])
            all_latents.append(latents)
            velocities.append(velocity)
        return {
            "latents": torch.stack(all_latents, dim=1),  # (B, num_steps + 1, C, H, W)
            "velocities": torch.stack(velocities, dim=1),  # (B, num_steps, C, H, W)
        }

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """Sample images. Return them, and their states and velocities to train on."""
        embeds, pooled_embeds = self.pipeline.encode_prompt(
            prompts, self.max_sequence_length
        )
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
        })
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "prompt_ids": prompt_ids(prompts).to(self.device),  # prompt groups
            "latents": out["latents"],
            "velocities": out["velocities"],
        }
        images = self.pipeline.decode(out["latents"][:, -1])
        return {"samples": samples, "responses": images}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images with the Euler ODE sampler, without CFG."""
        if guidance_scale != 1.0:
            raise ValueError("❌ DiffusionART samples only without CFG (scale 1.0)")
        embeds, pooled_embeds = self.pipeline.encode_prompt(
            prompts, self.max_sequence_length
        )
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "num_steps": num_steps,
        })
        return self.pipeline.decode(out["latents"][:, -1])

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """Subtract each prompt group's mean. A group with equal rewards gets zeros."""
        prompts = np.array(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards)
        for prompt in np.unique(prompts):
            group = rewards[prompts == prompt]
            if group.max() > group.min():
                advantages[prompts == prompt] = group - group.mean()
        return advantages

    @torch.no_grad()
    def compute_guidance(self, samples: dict) -> torch.Tensor:
        """
        Return the guidance G(c) of each sample's prompt group; see paper Eq. 5.

        G(c) is the covariance of the centered rewards and the clean latents of the
        group on all ranks, divided by its RMS. Thus the 1 / K of the covariance does
        not matter.
        """
        keys = torch.unique(gather_tensor(samples["prompt_ids"]))
        groups = torch.searchsorted(keys, samples["prompt_ids"])
        x0 = samples["latents"][:, -1]
        weights = samples["advantages"].float().view(-1, *[1] * (x0.ndim - 1))
        cov = torch.zeros((len(keys), *x0.shape[1:]), device=x0.device)
        cov.index_add_(0, groups, weights * x0)
        dist.all_reduce(cov)
        rms = cov.square().mean(dim=tuple(range(1, x0.ndim)), keepdim=True).sqrt()
        return torch.where(rms > 0, cov / rms, 0.0)[groups]
