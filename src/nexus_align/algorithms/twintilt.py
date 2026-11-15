"""
TwinTilt algorithm: fit one preference correction per group of rollouts, and regress
the rollout velocities onto the corrected targets.

TwinTilt extends FDFO from image pairs to groups of K images. For each group, it fits a
correction u in the span of the centered endpoints, so that the softmax of the endpoints
along u matches the tilted rewards q = softmax(eta * a). Every member and timestep of
the group then trains toward the target v - u_scale * u.

References:
    - FDFO:
        - Paper: Finite Difference Flow Optimization for RL Post-Training of
          Text-to-Image Models (arXiv 2026)
        - Official code: https://github.com/NVlabs/finite-difference-flow-optimization
            - fdfo/sampling.py: edm_step, sample_image_pairs
            - fdfo/train_loop.py: predict_v, the KL to the base model of train_fdfo
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private, twintilt-twintilt.zip)
            - src/nexus_align/algorithms/twintilt.py: TwinTilt.sample, build_targets,
              loss_from_predictions
            - src/nexus_align/algorithms/components/twintilt_projection.py:
              prepare_exact_projection, solve_prepared_exact_projection
            - src/nexus_align/algorithms/components/twintilt_sampling.py: sample_euler
            - src/nexus_align/trainers/twintilt_trainer.py: the reward z-scores of
              TwinTiltTrainer.sample_epoch
            - src/nexus_align/rewards/twintilt_multi.py:
              TwinTiltMultiReward.training_rewards
    - AWM:
        - Official code: https://github.com/scxue/advantage_weighted_matching
            - advantage_weighted_matching/scripts/train_sd3_awm.py: the base KL with
              CFG in the training loop of main
"""

import math

import torch
import torch.distributed as dist

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.engine.distributed import gather_tensor
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def group_zscore(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Standardize the rewards of each group with the population std plus 1e-4."""
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(1, keepdim=True)
    std = grouped.std(1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + 1e-4)).flatten()


def centered_basis(k: int, device: torch.device) -> torch.Tensor:
    """Return an orthonormal (K, K - 1) basis of the vectors with a zero mean."""
    basis = torch.zeros(k, k - 1, device=device, dtype=torch.float64)
    for j in range(k - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        basis[: j + 1, j] = 1 / scale
        basis[j + 1, j] = -(j + 1) / scale
    return basis


@torch.no_grad()
def solve_projection(
    points: torch.Tensor,
    advantages: torch.Tensor,
    eta: float,
    iterations: int,
    ridge_scale: float,
    ridge_eps: float,
) -> torch.Tensor:
    """
    Solve the correction u of each group with a fixed number of Newton updates, in fp64.

    For the centered endpoints D (K, n) of a group, u minimizes
    K * n * KL(q || softmax(D @ u / n)) + ridge * |u|^2 / 2, where q = softmax(eta * a)
    and ridge = ridge_scale * mean(diag(D @ D.T / n)) + ridge_eps. The solver works in
    the coordinates y of the eigenvectors of D @ D.T / n with nonzero eigenvalues.
    """
    # 1. Get the geometry of each group: points is (G, K, ...), and d is (G, K, n).
    d = points.double().flatten(2)
    d = d - d.mean(1, keepdim=True)
    groups, k, n = d.shape
    gram = torch.bmm(d, d.transpose(1, 2)) / n
    ridge = ridge_scale * gram.diagonal(dim1=1, dim2=2).mean(1) + ridge_eps
    center = centered_basis(k, d.device)
    reduced = center.T @ gram @ center
    eigenvalues, eigenvectors = torch.linalg.eigh(
        0.5 * (reduced + reduced.transpose(1, 2))
    )
    # Drop the directions that rounding alone makes nonzero.
    cutoff = 64 * torch.finfo(d.dtype).eps * eigenvalues.abs().amax(1, keepdim=True)
    keep = eigenvalues > cutoff
    roots = torch.where(
        keep, eigenvalues.clamp_min(0).sqrt(), torch.ones_like(eigenvalues)
    )
    vectors = center @ eigenvectors
    logits_map = vectors * (roots * keep).unsqueeze(1)  # the logits D @ u / n of y
    coefficient_map = vectors * (keep / roots).unsqueeze(1)  # u = D.T @ (map @ y)
    lt = logits_map.transpose(1, 2)

    # 2. Tilt the centered advantages.
    a = advantages.double()
    a = a - a.mean(1, keepdim=True)
    logq = (eta * a).log_softmax(1)
    q = logq.exp()

    def logits(y: torch.Tensor) -> torch.Tensor:
        return torch.bmm(logits_map, y.unsqueeze(2)).squeeze(2)

    def objective(y: torch.Tensor) -> torch.Tensor:
        """Return the objective divided by K * n."""
        logp = logits(y).log_softmax(1)
        return (q * (logq - logp)).sum(1) + ridge / (2 * k) * y.square().sum(1)

    # 3. Do the Newton updates. Each group backtracks with the Armijo rule on its own.
    y = d.new_zeros(groups, k - 1)
    eye = torch.eye(k - 1, device=d.device, dtype=d.dtype)
    for _ in range(iterations):
        p = logits(y).softmax(1)
        gradient = k * torch.bmm(lt, (p - q).unsqueeze(2)).squeeze(2)
        gradient = gradient + ridge[:, None] * y
        mean = torch.bmm(lt, p.unsqueeze(2)).squeeze(2)
        covariance = torch.bmm(lt, p[:, :, None] * logits_map)
        covariance = covariance - mean[:, :, None] * mean[:, None, :]
        hessian = k * covariance + ridge[:, None, None] * eye
        delta = torch.linalg.solve(hessian, gradient.unsqueeze(2)).squeeze(2)
        slope = (gradient * delta).sum(1) / k
        before = objective(y)
        slack = 8 * torch.finfo(d.dtype).eps * before.abs().clamp_min(1.0)
        rate = d.new_ones(groups)
        accepted = torch.zeros(groups, device=d.device, dtype=torch.bool)
        for _ in range(50):
            value = objective(y - rate[:, None] * delta)
            accepted |= torch.isfinite(value) & (
                value <= before - 1e-4 * rate * slope + slack
            )
            if accepted.all():
                break
            rate = torch.where(accepted, rate, rate * 0.5)
        else:
            raise RuntimeError("❌ The line search of the TwinTilt projection failed")
        y = y - rate[:, None] * delta

    # 4. Map y back to the latent space.
    coefficients = torch.bmm(coefficient_map, y.unsqueeze(2))
    u = torch.bmm(d.transpose(1, 2), coefficients).squeeze(2)
    return u.reshape(groups, *points.shape[2:])


def churn_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    next_sigma: float,
    churn: float,
) -> torch.Tensor:
    """
    Do one Euler step from sigma to next_sigma with churn, as FDFO's edm_step does.

    The step overshoots to r < next_sigma, then adds fresh noise that brings the noise
    level back to next_sigma. Without churn, and in the last step to sigma 0, it is a
    plain Euler step.
    """
    if churn == 0 or next_sigma == 0:
        return latents + (next_sigma - sigma) * velocity
    gamma = torch.tensor(churn)  # fp32 coefficients, as in FDFO
    r = next_sigma / (-gamma * next_sigma + gamma + 1)
    overshoot = latents + (r - sigma) * velocity
    noise_scale = r * ((gamma + 1) ** 2 - 1) ** 0.5 / (gamma * r + 1)
    scale = 1 / (gamma * r + 1)
    return scale * overshoot + noise_scale * torch.randn_like(overshoot)


class TwinTilt(BaseAlgorithm):
    """TwinTilt training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Sampling configs.
        self.resolution = cfg_algorithm.get("resolution", 512)  # image resolution
        self.num_steps = cfg_algorithm.get("num_steps", 10)  # rollout sampling steps of the Euler solver
        self.churn_weight = cfg_algorithm.get("churn_weight", 0.04)  # churn gamma of the rollout noise; 0 gives the ODE
        self.optimization_level = cfg_algorithm.get("optimization_level", "trajectory")  # trajectory: one noise per group; distribution: one per image
        self.max_sequence_length = cfg_algorithm.get("max_sequence_length", 128)  # T5 prompt tokens
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "fp16")  # transformer compute dtype

        # Projection configs.
        self.group_size = cfg_algorithm.get("group_size", 24)  # images per prompt, K
        self.group_zscore = cfg_algorithm.get("group_zscore", True)  # z-score a single reward in each group; false keeps the raw scores
        self.eta = cfg_algorithm.get("eta", 1.0)  # tilt strength; q = softmax(eta * advantages)
        self.u_scale = cfg_algorithm.get("u_scale", 0.1)  # scale of the solved correction u
        self.projection_iterations = cfg_algorithm.get("projection_iterations", 5)  # Newton updates of the projection
        self.ridge_scale = cfg_algorithm.get("ridge_scale", 0.001)  # ridge = ridge_scale * mean Gram diagonal + ridge_eps
        self.ridge_eps = cfg_algorithm.get("ridge_eps", 1.0e-6)  # lowest ridge of the projection

        # Policy loss configs.
        self.kl_weight = cfg_algorithm.get("kl_weight", 0.004)  # coefficient of the velocity KL to the base model
        self.kl_cfg_scale = cfg_algorithm.get("kl_cfg_scale", 1.0)  # CFG scale of the base model in this KL; 1 gives no CFG

        self.pipeline = None
        self.scheduler = None
        self.neg_embeds = None  # empty-prompt embeddings for CFG
        # Forward passes per training batch, one per sampling timestep.
        self.num_micro_steps = self.num_steps

    def prepare(self, model, device, pipeline) -> None:
        """Bind the transformer and device, and the pipeline with the scheduler."""
        super().prepare(model, device)
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler

    def predict(
        self, transformer, latents, timesteps, embeds, pooled_embeds, guidance_scale=1.0
    ):
        """
        Predict the velocity under autocast.

        With CFG, embeds holds the negative prompts first, as in the diffusers pipeline.
        """
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents, timesteps = torch.cat([latents] * 2), torch.cat([timesteps] * 2)
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            velocity = transformer(
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
            self.neg_embeds = self.pipeline.encode_prompt([""], self.max_sequence_length)
        neg_embeds, neg_pooled_embeds = self.neg_embeds
        embeds = torch.cat([neg_embeds.expand_as(embeds), embeds])
        pooled_embeds = torch.cat(
            [neg_pooled_embeds.expand_as(pooled_embeds), pooled_embeds]
        )
        return embeds, pooled_embeds

    def forward(self, data):
        """Regress the velocity at the data["step"]-th timestep onto its target."""
        j = data["step"]
        args = (data["xs"][:, j], data["timesteps"][:, j])
        args += (data["prompt_embeds"], data["pooled_prompt_embeds"])
        pred = self.predict(self.model, *args)
        # The base model can use CFG in the KL, as in AWM.
        with torch.no_grad(), self.model.disable_adapter():
            if self.kl_cfg_scale > 1.0:
                args = args[:2] + self.negative_first(*args[2:])
            base_pred = self.predict(self.model, *args, self.kl_cfg_scale)

        # Every member and timestep of a group moves by the same correction.
        target = data["velocities"][:, j].float() - data["displacement"]
        dims = tuple(range(1, pred.ndim))
        policy_loss = (pred.float() - target).square().mean(dims).mean()
        # Subtract in the autocast dtype and square in fp32, as in TwinTilt.
        kl_loss = (pred - base_pred).float().square().mean(dims).mean()
        loss = policy_loss + self.kl_weight * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "loss": loss.detach(),
        }
        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def sample(self, data):
        """
        Sample from data["latents"] with the Euler solver and churn, as FDFO does.

        Return the final latents, and the inputs, velocities, and timesteps of the model
        calls.
        """
        latents = data["latents"]
        transformer = data.get("transformer", self.model)
        num_steps = data.get("num_steps", self.num_steps)
        churn = data.get("churn_weight", self.churn_weight)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps, sigmas = self.scheduler.timesteps, self.scheduler.sigmas.tolist()
        xs, velocities = [], []
        for i, t in enumerate(timesteps):
            velocity = self.predict(
                transformer,
                latents,
                t.expand(latents.shape[0]),
                data["prompt_embeds"],
                data["pooled_prompt_embeds"],
                data.get("guidance_scale", 1.0),
            )
            xs.append(latents.float())
            velocities.append(velocity)
            latents = churn_step(
                latents.float(), velocity.float(), sigmas[i], sigmas[i + 1], churn
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

    def initial_noise(
        self, num_images: int, dtype: torch.dtype, group_ids: list[int] | None = None
    ) -> torch.Tensor:
        """
        Draw the initial noise in the prompt dtype, as the diffusers pipeline does.

        If group_ids is given, each group ID seeds the noise of its images. Then the
        images of a group share one noise, also on other ranks.
        """
        latent_size = self.resolution // 8
        shape = (self.model.config.in_channels, latent_size, latent_size)
        if group_ids is None:
            return torch.randn((num_images, *shape), device=self.device, dtype=dtype)
        noise = [
            torch.randn(shape, generator=torch.Generator().manual_seed(g))
            for g in group_ids
        ]
        return torch.stack(noise).to(self.device, dtype)

    @torch.no_grad()
    def rollout(self, prompts: list[str], group_ids: list[int]) -> dict:
        """
        Sample one image per prompt with the EMA, which tracks the policy.

        The images of one group have the same group ID, on all ranks.
        """
        embeds, pooled_embeds = self.pipeline.encode_prompt(
            prompts, self.max_sequence_length
        )

        # A trajectory group shares one noise, and churn gives each image its own copy.
        if self.optimization_level == "trajectory":
            latents = self.initial_noise(len(prompts), embeds.dtype, group_ids)
            if self.churn_weight > 0:
                gamma = torch.tensor(self.churn_weight)  # fp32 coefficients, as in FDFO
                scale = 1 / (gamma + 1)
                noise_scale = ((gamma + 1).square() - 1).sqrt() / (gamma + 1)
                noise = torch.randn_like(latents)
                latents = scale * latents.float() + noise_scale * noise.float()
        else:
            latents = self.initial_noise(len(prompts), embeds.dtype)

        transformer = self.pipeline.ema if self.pipeline.ema is not None else self.model
        out = self.sample({
            "latents": latents,
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "transformer": transformer,
        })
        samples = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "timesteps": out["timesteps"].repeat(len(embeds), 1),
            "xs": out["xs"],
            "velocities": out["velocities"],
            "latents": out["latents"],  # the endpoints, for the projection
            "group_ids": torch.tensor(group_ids, device=self.device),
        }
        return {"samples": samples, "responses": self.decode(out["latents"])}

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

    def compute_advantages(self, scores: dict, reward) -> torch.Tensor:
        """
        Get the training reward of each image. The images of a group are in a row.

        - A single reward gives its group z-scores if group_zscore, else its raw scores.
        - A multi-reward takes the weighted mean of the group z-scores of its calibrated
          components.
        """
        k = self.group_size
        weights = getattr(reward, "weights", None)
        if weights is None:
            rewards = scores[reward.name]
            return group_zscore(rewards, k) if self.group_zscore else rewards
        total = sum(
            weight * group_zscore(scores[f"{key}_calibrated"], k)
            for key, weight in weights.items()
        )
        return total / sum(weights.values())

    @torch.no_grad()
    def build_targets(self, samples: dict, scores: dict, reward) -> tuple[dict, dict]:
        """
        Get the advantages and the correction of each image, from the groups of all
        ranks. scores holds the reward scores of the images of this rank.
        """
        # 1. Gather the images of all ranks, and sort them by group. Every rank gets
        # the same order, and the images of a group are then in a row.
        k = self.group_size
        group_ids = gather_tensor(samples.pop("group_ids"))
        order = torch.argsort(group_ids, stable=True)
        latents = gather_tensor(samples.pop("latents"))[order]
        # Gather only the scores that every image has, which the advantages need.
        weights = getattr(reward, "weights", {})
        keys = [reward.name] + [f"{key}_calibrated" for key in weights]
        scores = {key: gather_tensor(scores[key])[order] for key in keys}

        # 2. Solve the correction of each group.
        advantages = self.compute_advantages(scores, reward)
        u = solve_projection(
            latents.reshape(-1, k, *latents.shape[1:]),
            advantages.reshape(-1, k),
            self.eta,
            self.projection_iterations,
            self.ridge_scale,
            self.ridge_eps,
        )
        displacement = (u * self.u_scale).float()

        # 3. Keep the advantages and the corrections of the images of this rank.
        size = len(samples["xs"])
        rank = dist.get_rank()
        position = torch.argsort(order)[rank * size: (rank + 1) * size]
        samples["advantages"] = advantages[position]
        samples["displacement"] = displacement[position // k]
        metrics = {
            "projection_rms": displacement.square().mean().sqrt().item(),
            "advantages_abs_mean": advantages.abs().mean().item(),
            "reward_std_mean": (
                scores[reward.name].reshape(-1, k).std(1, unbiased=False).mean().item()
            ),
        }
        return samples, metrics
