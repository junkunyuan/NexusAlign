"""DiffusionNFT algorithm: negative-aware fine-tuning on the forward process.

References:
    - DiffusionNFT:
        - Paper: DiffusionNFT: Online Diffusion Reinforcement with Forward Process
        - Official code: https://github.com/NVlabs/DiffusionNFT
"""

from contextlib import contextmanager

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def dpm2_step(
    x0_preds: list[torch.Tensor], sample: torch.Tensor, sigmas: torch.Tensor, i: int
) -> torch.Tensor:
    """
    One step from sigmas[i] to sigmas[i + 1] of the 2nd-order multistep DPM-Solver.

    x0_preds holds the x0 predictions up to step i; the first and the last steps are
    1st-order (DDIM), as in the official deterministic dpm2 solver.
    """
    sigmas = sigmas.to(torch.float64)
    t, s = sigmas[i + 1], sigmas[i]
    if i == 0 or i == len(sigmas) - 2:
        noise_pred = (sample - (1 - s) * x0_preds[-1]) / s
        return (1 - t) * x0_preds[-1] + torch.sqrt(t**2) * noise_pred

    # The flow schedule has alpha = 1 - sigma; lambda is the half log-SNR.
    sigma_t, sigma_s0, sigma_s1 = t, s, sigmas[i - 1]
    alpha_t, alpha_s0, alpha_s1 = 1 - sigma_t, 1 - sigma_s0, 1 - sigma_s1
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    m0, m1 = x0_preds[-1], x0_preds[-2]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)
    return (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )


def adaptive_x0_loss(x0_pred: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """x0 regression divided by its stop-gradient mean absolute error, per sample."""
    dims = tuple(range(1, x0.ndim))
    with torch.no_grad():
        weight = (
            torch.abs(x0_pred.double() - x0.double())
            .mean(dim=dims, keepdim=True)
            .clip(min=0.00001)
        )
    return ((x0_pred - x0) ** 2 / weight).mean(dim=dims)


class DiffusionNFT(BaseAlgorithm):
    """DiffusionNFT training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 10)  # rollout sampling steps
        self.solver = cfg_algorithm.get("solver", "dpm2")  # rollout sampler: dpm2 (2nd-order DPM-Solver) or flow (Euler), both ODE
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Advantage configs.
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt, K
        self.global_std = cfg_algorithm.get("global_std", True)  # divide by the std of all rewards instead of the group's

        # Policy loss configs.
        self.timestep_fraction = cfg_algorithm.get("timestep_fraction", 0.99)  # train on int(num_steps * fraction) sampling timesteps of each image
        self.adv_clip_max = cfg_algorithm.get("adv_clip_max", 5.0)  # optimality r = 0.5 + 0.5 * clip(advantage / max, -1, 1)
        self.beta = cfg_algorithm.get("beta", 1.0)  # implicit policy coefficient beta; see paper Eq. (5)
        self.kl_beta = cfg_algorithm.get("kl_beta", 1.0e-4)  # coefficient of the velocity KL to the reference policy

        # Soft update configs.
        self.eta_rate = cfg_algorithm.get("eta_rate", 0.001)  # old policy decay eta = min(eta_rate * (step - eta_delay), eta_max)
        self.eta_max = cfg_algorithm.get("eta_max", 0.5)  # max eta
        self.eta_delay = cfg_algorithm.get("eta_delay", 0)  # optimizer steps with eta = 0 (on-policy) before eta grows

        self.pipeline = None
        self.scheduler = None
        self.params = None  # trained weights
        self.old_params = None  # weights of the sampling (old) policy
        # Forward passes per training batch, one per trained timestep.
        self.num_micro_steps = int(self.num_steps * self.timestep_fraction)

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        # Local shards of the trained params, as sharded params are slow to dispatch.
        self.params = [
            (p.to_local() if isinstance(p, DTensor) else p).detach()
            for p in model.parameters() if p.requires_grad
        ]

    @contextmanager
    def old_policy(self):
        """Swap in the old policy's weights; it starts as a copy of the trained ones."""
        if self.old_params is None:
            self.old_params = [p.detach().clone() for p in self.params]
        current = [p.detach().clone() for p in self.params]
        torch._foreach_copy_(self.params, self.old_params)
        try:
            yield
        finally:
            torch._foreach_copy_(self.params, current)

    def reset_old_policy(self) -> None:
        """Restart the old policy as a copy of the trained weights at its next use."""
        self.old_params = None

    @torch.no_grad()
    def update_old_policy(self, epoch: int, step: int) -> None:
        """Soft-update the old policy toward the trained one; see paper Sec. 3.3."""
        if self.old_params is None:
            return
        eta = 0.0 if step < self.eta_delay else min(
            (step - self.eta_delay) * self.eta_rate, self.eta_max
        )
        for p, old in zip(self.params, self.old_params):
            old.copy_(old * eta + p.detach().clone() * (1.0 - eta))

    def predict(self, latents, sigma, embeds, pooled_embeds, guidance_scale):
        """Sampling velocity; the timestep is truncated to an integer, as officially."""
        do_cfg = guidance_scale > 1.0
        model_input = torch.cat([latents] * 2) if do_cfg else latents
        batch_size = model_input.shape[0]
        timesteps = torch.full(
            [batch_size], sigma * 1000, device=latents.device, dtype=torch.long
        )
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            noise_pred = self.model(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        noise_pred = noise_pred.to(embeds.dtype)
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred

    def forward(self, data):
        """NFT loss of one batch at its data["step"]-th timestep; see paper Eq. (5)."""
        # 1. Noise the clean latents at this image's timestep of the step.
        j = data["step"]
        x0, timesteps = data["latents"], data["timesteps"][:, j]
        t = (timesteps / 1000.0).view(-1, *([1] * (x0.ndim - 1)))
        noise = torch.randn_like(x0.float())
        xt = (1 - t) * x0 + t * noise

        # 2. Velocities of the old, trained, and reference (LoRA disabled) policies.
        inputs = {
            "hidden_states": xt,
            "timestep": timesteps,
            "encoder_hidden_states": data["prompt_embeds"],
            "pooled_projections": data["pooled_prompt_embeds"],
            "return_dict": False,
        }
        # The old policy runs in its own autocast region, as autocast caches the fp16
        # casts of trained weights within a region, which would keep the old ones.
        with torch.no_grad(), self.old_policy():
            with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
                old_pred = self.model(**inputs)[0]
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            pred = self.model(**inputs)[0]
            with torch.no_grad(), self.model.disable_adapter():
                ref_pred = self.model(**inputs)[0]

        # 3. Optimality-weighted losses of the implicit positive and negative policies.
        adv_clip_max = self.adv_clip_max
        advantages = torch.clamp(data["advantages"], -adv_clip_max, adv_clip_max)
        r = torch.clamp((advantages / adv_clip_max) / 2.0 + 0.5, 0, 1)
        positive_pred = self.beta * pred + (1 - self.beta) * old_pred
        negative_pred = (1.0 + self.beta) * old_pred - self.beta * pred
        positive_loss = adaptive_x0_loss(xt - t * positive_pred, x0)
        negative_loss = adaptive_x0_loss(xt - t * negative_pred, x0)
        beta = self.beta
        policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
        policy_loss = (policy_loss * adv_clip_max).mean()

        # 4. KL to the reference policy as the squared velocity difference.
        dims = tuple(range(1, x0.ndim))
        kl_loss = torch.mean(((pred - ref_pred) ** 2).mean(dim=dims))
        loss = policy_loss + self.kl_beta * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "old_deviate": torch.mean((pred - old_pred) ** 2).detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample latents as the official sampler does, with fp16 latents between steps.

        solver is "dpm2" for rollouts, or "flow" (Euler) for evaluation by default.
        """
        # 1. Prepare prompts, with negative prompts first for CFG.
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        guidance_scale = data.get("guidance_scale", 1.0)
        num_steps = data.get("num_steps", self.num_steps)
        solver = data.get("solver", "flow")
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
        )

        # 3. Denoise, stepping in fp32 and storing latents in the prompt dtype.
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps, sigmas = self.scheduler.timesteps, self.scheduler.sigmas.float()
        all_latents, x0_preds = [latents], []
        for i in range(num_steps):
            noise_pred = self.predict(
                latents, sigmas[i], embeds, pooled_embeds, guidance_scale
            )
            sample, model_output = latents.float(), noise_pred.float()
            if solver == "flow":
                latents = sample + model_output * (sigmas[i + 1] - sigmas[i])
            else:
                x0_preds.append(sample - sigmas[i] * model_output)
                latents = dpm2_step(x0_preds, sample, sigmas, i)
            latents = latents.to(embeds.dtype)
            all_latents.append(latents)

        return {
            "latents": torch.stack(all_latents, dim=1),  # (B, num_steps + 1, C, H, W)
            "timesteps": timesteps.repeat(batch_size, 1),  # (B, num_steps)
        }

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images under autocast, as the official pipeline does."""
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            return self.pipeline.decode(latents)

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """Sample images with the old policy; return their clean latents and images."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        with self.old_policy():
            out = self.sample({
                "prompt_embeds": embeds,
                "pooled_prompt_embeds": pooled_embeds,
                "solver": self.solver,
            })

        # Each image trains on its own sampling timesteps, in a random order.
        batch_size, num_steps = out["timesteps"].shape
        order = torch.stack(
            [torch.randperm(num_steps, device=self.device) for _ in range(batch_size)]
        )
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "timesteps": out["timesteps"].gather(1, order),
            "latents": out["latents"][:, -1],  # clean latents
        }
        return {"samples": samples, "responses": self.decode(out["latents"][:, -1])}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Evaluation images of the Euler ODE sampler at the given CFG scale."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        data = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
        }
        if guidance_scale > 1.0:
            neg_embeds, neg_pooled_embeds = self.pipeline.encode_prompt([""], max_length)
            data["neg_prompt_embeds"] = neg_embeds.repeat(len(prompts), 1, 1)
            data["neg_pooled_prompt_embeds"] = neg_pooled_embeds.repeat(len(prompts), 1)
        out = self.sample(data)
        return self.decode(out["latents"][:, -1])

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """Subtract each prompt group's mean; std is global when global_std is set."""
        prompts = np.array(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards)
        for prompt in np.unique(prompts):
            group = rewards[prompts == prompt]
            mean = np.mean(group, axis=0, keepdims=True)
            std = rewards if self.global_std else group
            std = np.std(std, axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (group - mean) / std
        return advantages
