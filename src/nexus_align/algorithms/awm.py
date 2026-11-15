"""
AWM algorithm: the flow matching loss on rollouts, weighted by their advantages.

References:
    - AWM:
        - Paper: Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion
          Models (ICML 2026)
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py:
              compute_log_prob_awm, for the generalized Huber log-likelihood; the
              training loop of main, for the policy loss, the base KL with CFG, the
              EMA KL, and the soft update of the EMA reference
            - advantage_weighted_matching/flow_grpo/prompt_noise.py:
              prompt_training_seed, shared_prompt_training_inputs
            - advantage_weighted_matching/flow_grpo/diffusers_patch/sa_solver.py:
              SASolver.sample, adams_bashforth_update_few_steps,
              adams_moulton_update_few_steps
            - advantage_weighted_matching/flow_grpo/diffusers_patch/
              sd3_pipeline_with_logprob.py: pipeline_with_logprob, for the SA-Solver
              rollout and the Euler evaluation
"""

import json
import hashlib
from contextlib import contextmanager

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.algorithms.flow_grpo import group_advantages
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def prompt_seed(epoch: int, prompt: str) -> int:
    """Get a 63-bit seed from the epoch and the prompt text, the same on all ranks."""
    payload = json.dumps(
        ["prompt-training-v1", epoch, prompt], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def sa_step(
    x: torch.Tensor,
    x0_preds: list[torch.Tensor],
    lambdas: list[torch.Tensor],
    t_prev: torch.Tensor,
    t: torch.Tensor,
    tau: float,
    noise: torch.Tensor,
    corrector: bool,
) -> torch.Tensor:
    """
    Do one SA-Solver step from t_prev to t in data prediction, of order len(x0_preds).

    x0_preds and lambdas hold the data predictions and their half log-SNR, the newest
    first. The predictor uses the predictions up to t_prev, and the corrector also the
    one at t. An order-2 step adds the few-step correction of the official code.
    """
    lambda_prev = torch.log(1 - t_prev) - torch.log(t_prev)
    lambda_t = torch.log(1 - t) - torch.log(t)
    h = lambda_t - lambda_prev

    # Integrals of exp(c * u) * u^k over [lambda_prev, lambda_t], with c = 1 + tau^2.
    c = 1 + tau**2
    start, end = c * lambda_prev, c * lambda_t
    decay = torch.exp(-(end - start))
    exp_int0 = torch.exp(end) * (1 - decay) / c
    exp_int1 = torch.exp(end) * ((end - 1) - (start - 1) * decay) / c**2

    # Integrate the Lagrange polynomial through the data predictions.
    if len(x0_preds) == 1:
        coefficients = [exp_int0]
    else:
        l0, l1 = lambdas
        coefficients = [
            (exp_int1 - l1 * exp_int0) / (l0 - l1),
            (exp_int1 - l0 * exp_int0) / (l1 - l0),
        ]
        rest = h * c - 1 + torch.exp(-c * h)
        if corrector:
            extra = torch.exp(c * lambda_t) * (h / 2 - rest / (c**2 * h))
        else:
            extra = torch.exp(c * lambda_t) * (h**2 / 2 - rest / c**2) / (l0 - l1)
        coefficients = [coefficients[0] + extra, coefficients[1] - extra]

    x_t = torch.exp(-tau**2 * h) * (t / t_prev) * x.float()
    scale = c * t * torch.exp(-tau**2 * lambda_t)
    for coefficient, x0_pred in zip(coefficients, x0_preds):
        x_t = x_t + scale * coefficient * x0_pred.float()
    return x_t + t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise.float()


class AWM(BaseAlgorithm):
    """AWM training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 14)  # rollout sampling steps of the SA-Solver
        self.noise_level = cfg_algorithm.get("noise_level", 0.4)  # SA-Solver noise tau for t in [0.3, 0.9]; 0 gives the ODE
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Advantage configs.
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt, G
        self.adv_clip_max = cfg_algorithm.get("adv_clip_max", 5.0)  # clip the advantages to [-adv_clip_max, adv_clip_max]
        self.advantage_max = cfg_algorithm.get("advantage_max", 1.0)  # then scale them to [-advantage_max, advantage_max]

        # Policy loss configs.
        self.num_train_timesteps = cfg_algorithm.get("num_train_timesteps", 6)  # Monte Carlo timesteps T of each image: the first sampling step, plus stratified ones
        self.timestep_fraction = cfg_algorithm.get("timestep_fraction", 0.9)  # draw the stratified timesteps from steps 1 to int(num_steps * fraction)
        self.ghuber_power = cfg_algorithm.get("ghuber_power", 0.5)  # power p of the generalized Huber log-likelihood; 1 gives w(t) = t of paper Eq. 15

        # KL configs.
        self.kl_beta = cfg_algorithm.get("kl_beta", 0.003)  # coefficient of the velocity KL to the base model
        self.kl_cfg_scale = cfg_algorithm.get("kl_cfg_scale", 4.5)  # CFG scale of the base model in this KL
        self.ref_kl_beta = cfg_algorithm.get("ref_kl_beta", 1.0)  # coefficient beta of the velocity KL to the EMA reference; see paper Eq. 16
        self.ref_decay_rate = cfg_algorithm.get("ref_decay_rate", 0.001)  # EMA reference decay = min(ref_decay_rate * step, ref_decay_max)
        self.ref_decay_max = cfg_algorithm.get("ref_decay_max", 0.3)  # highest decay of the EMA reference

        self.pipeline = None
        self.scheduler = None
        self.params = None  # trained weights
        self.ref_params = None  # weights of the EMA reference
        self.neg_embeds = None  # empty-prompt embeddings for the CFG of the base model
        # Forward passes per training batch, one per trained timestep.
        self.num_micro_steps = self.num_train_timesteps

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        # Keep plain local shards of the trained params. Ops on a DTensor cost CPU time.
        self.params = [
            (p.to_local() if isinstance(p, DTensor) else p).detach()
            for p in model.parameters() if p.requires_grad
        ]

    @contextmanager
    def reference(self):
        """Swap in the EMA reference weights, which start as the trained ones."""
        if self.ref_params is None:
            self.ref_params = [p.detach().clone() for p in self.params]
        current = [p.detach().clone() for p in self.params]
        torch._foreach_copy_(self.params, self.ref_params)
        try:
            yield
        finally:
            torch._foreach_copy_(self.params, current)

    @torch.no_grad()
    def update_reference(self, step: int) -> None:
        """Soft-update the EMA reference toward the trained weights after each epoch."""
        if self.ref_params is None:
            return
        decay = min(self.ref_decay_rate * step, self.ref_decay_max)
        for p, ref in zip(self.params, self.ref_params):
            ref.copy_(ref * decay + p * (1.0 - decay))

    def predict(self, latents, timesteps, embeds, pooled_embeds, guidance_scale):
        """Predict the velocity at float timesteps, with CFG if guidance_scale > 1."""
        do_cfg = guidance_scale > 1.0
        model_input = torch.cat([latents] * 2) if do_cfg else latents
        timesteps = torch.cat([timesteps] * 2) if do_cfg else timesteps
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            noise_pred = self.model(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.float().chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        return noise_pred

    def forward(self, data):
        """Calculate the AWM loss at the data["step"]-th training timestep."""
        # 1. Add the noise of this step to the clean latents at its timestep.
        j = data["step"]
        x0, noise = data["latents"], data["noise"][:, j]
        timesteps = data["timesteps"][:, j]
        t = (timesteps / 1000.0).view(-1, *([1] * (x0.ndim - 1)))
        xt = (1 - t) * x0 + t * noise

        # 2. Predict the velocities of the EMA reference, the policy, and the base model
        # (LoRA disabled) with CFG.
        inputs = (xt, timesteps, data["prompt_embeds"], data["pooled_prompt_embeds"])
        # The reference runs in its own autocast region. Autocast caches the fp16 casts
        # of the trained weights in a region, so a shared region keeps the old casts.
        with torch.no_grad(), self.reference():
            ref_pred = self.predict(*inputs, guidance_scale=1.0)
        pred = self.predict(*inputs, guidance_scale=1.0)
        with torch.no_grad(), self.model.disable_adapter():
            batch_size = x0.shape[0]
            neg_embeds, neg_pooled_embeds = self.neg_embeds
            base_pred = self.predict(
                xt,
                timesteps,
                torch.cat([neg_embeds.expand(batch_size, -1, -1), inputs[2]]),
                torch.cat([neg_pooled_embeds.expand(batch_size, -1), inputs[3]]),
                guidance_scale=self.kl_cfg_scale,
            )

        # 3. Weight the gradient of each image's log-likelihood surrogate by its
        # advantage. On policy, the ratio is 1, so the PPO clip has no effect; see paper
        # Eq. 15 and 17.
        dims = tuple(range(1, x0.ndim))
        pred = pred.float()
        mse = ((pred - (noise.float() - x0.float())) ** 2).mean(dim=dims)
        p = self.ghuber_power
        log_prob = -((mse + 1e-10) ** p - 1e-10**p) * t.flatten() / p
        ratio = torch.exp(log_prob - log_prob.detach())
        adv_clip_max = self.adv_clip_max
        advantages = torch.clamp(data["advantages"], -adv_clip_max, adv_clip_max)
        advantages = advantages / adv_clip_max * self.advantage_max
        policy_loss = torch.mean(-advantages * ratio)

        # 4. Use the squared velocity differences as the KLs to the base model and to
        # the EMA reference; see paper Eq. 16.
        kl_loss = torch.mean(((pred - base_pred.float()) ** 2).mean(dim=dims))
        ref_kl_loss = torch.mean(((pred - ref_pred.float()) ** 2).mean(dim=dims))
        loss = policy_loss + self.kl_beta * kl_loss + self.ref_kl_beta * ref_kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "ref_kl_loss": ref_kl_loss.detach(),
            "fm_loss": mse.mean().detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample latents as the official sampler does, with fp16 latents between steps.

        solver is "sa" (SA-Solver) for rollouts, or "euler" (ODE) for evaluation.
        """
        # 1. Prepare the prompts, with the negative prompts first for CFG.
        embeds, pooled_embeds = data["prompt_embeds"], data["pooled_prompt_embeds"]
        guidance_scale = data.get("guidance_scale", 1.0)
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
        )
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps, sigmas = self.scheduler.timesteps, self.scheduler.sigmas.float()

        def velocity(x, sigma):
            t = (sigma * 1000).expand(batch_size)  # the fp32 timesteps of the scheduler
            v = self.predict(x, t, embeds, pooled_embeds, guidance_scale)
            return v.to(embeds.dtype)

        # 3a. Denoise with the Euler ODE in fp32 steps.
        if data.get("solver", "euler") == "euler":
            for i in range(num_steps):
                v = velocity(latents, sigmas[i]).float()
                latents = latents.float() + v * (sigmas[i + 1] - sigmas[i])
                latents = latents.to(embeds.dtype)
            return {"latents": latents, "timesteps": timesteps.repeat(batch_size, 1)}

        # 3b. Denoise with the 2nd-order SA-Solver (PEC) on the official times: the
        # first is 1 - 1e-3, and the second last is half of the one before it. The first
        # step is 1st-order, and the last step returns the last data prediction.
        ts = sigmas.clone()
        ts[-2], ts[0] = ts[-3] / 2, 1 - 1e-3
        x0_pred = lambda x, t: (x.float() - t * velocity(x, t).float()).to(embeds.dtype)
        x0_preds = [x0_pred(latents, ts[0])]
        lambdas = [torch.log(1 - ts[0]) - torch.log(ts[0])]
        for i in range(1, num_steps):
            t, t_prev = ts[i], ts[i - 1]
            tau = self.noise_level if 0.3 <= t <= 0.9 else 0.0
            noise = torch.randn_like(latents)
            order = 1 if i == 1 else 2
            x_p = sa_step(
                latents, x0_preds[:order], lambdas[:order], t_prev, t, tau, noise, False
            ).to(embeds.dtype)
            x0_preds.insert(0, x0_pred(x_p, t))
            lambdas.insert(0, torch.log(1 - t) - torch.log(t))
            if i > 1:  # the first step has no corrector
                x_p = sa_step(
                    latents, x0_preds[:2], lambdas[:2], t_prev, t, tau, noise, True
                ).to(embeds.dtype)
            latents = x_p
            del x0_preds[2:], lambdas[2:]
        return {"latents": x0_preds[0], "timesteps": timesteps.repeat(batch_size, 1)}

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images with the fp32 VAE, as the official pipeline does."""
        return self.pipeline.decode(latents)

    @torch.no_grad()
    def training_inputs(
        self, prompts: list[str], epoch: int, timesteps: torch.Tensor, x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw the training timesteps and noise of each image from its prompt and epoch.

        All images of one prompt share them in an epoch, on all ranks. The timesteps are
        the first sampling step, plus one stratified step from each part of steps 1 to
        int(num_steps * timestep_fraction).
        """
        num = self.num_train_timesteps - 1
        end = int(self.num_steps * self.timestep_fraction)
        bounds = torch.linspace(1, end, num + 1, device=self.device)
        lower, upper = bounds[:-1].long(), bounds[1:].long()
        cache = {}
        for prompt in set(prompts):
            g = torch.Generator(device=self.device)
            g.manual_seed(prompt_seed(epoch, prompt))
            u = torch.rand(num, generator=g, device=self.device)
            stratified = lower + (u * (upper - lower)).long()
            index = torch.cat([lower.new_zeros(1), stratified])
            noise = torch.randn(
                (self.num_train_timesteps, *x0.shape[1:]),
                generator=g, device=self.device, dtype=x0.dtype,
            )
            cache[prompt] = index, noise
        train_timesteps = torch.stack(
            [timesteps[i, cache[prompt][0]] for i, prompt in enumerate(prompts)]
        )
        noise = torch.stack([cache[prompt][1] for prompt in prompts])
        return train_timesteps, noise

    @torch.no_grad()
    def rollout(self, prompts: list[str], epoch: int) -> dict:
        """Sample images with the SA-Solver. Return them and their training inputs."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        if self.neg_embeds is None:
            self.neg_embeds = self.pipeline.encode_prompt([""], max_length)
        out = self.sample({
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "solver": "sa",
        })
        x0 = out["latents"]
        timesteps, noise = self.training_inputs(prompts, epoch, out["timesteps"], x0)
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "timesteps": timesteps,  # (B, num_train_timesteps)
            "noise": noise,  # (B, num_train_timesteps, C, H, W)
            "latents": x0,  # clean latents
        }
        return {"samples": samples, "responses": self.decode(x0)}

    @torch.no_grad()
    def generate(self, prompts: list[str], num_steps: int, guidance_scale: float):
        """Make evaluation images with the Euler ODE sampler at the given CFG scale."""
        max_length = self.max_sequence_length
        embeds, pooled_embeds = self.pipeline.encode_prompt(prompts, max_length)
        data = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
        }
        if guidance_scale > 1.0:
            neg_embeds, neg_pooled = self.pipeline.encode_prompt([""], max_length)
            data["neg_prompt_embeds"] = neg_embeds.repeat(len(prompts), 1, 1)
            data["neg_pooled_prompt_embeds"] = neg_pooled.repeat(len(prompts), 1)
        out = self.sample(data)
        return self.decode(out["latents"])

    def compute_advantages(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        return group_advantages(prompts, rewards)
