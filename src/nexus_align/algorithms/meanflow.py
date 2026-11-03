"""
MeanFlow algorithm: mean-velocity loss with a JVP target, and one-step sampling.

References:
    - MeanFlow:
        - Paper: Mean Flows for One-step Generative Modeling (NeurIPS 2025)
        - Official code: https://github.com/Gsunshine/meanflow
            - meanflow.py: sample_tr, guidance_fn, cond_drop, forward, generate
        - Unofficial code: https://github.com/zhuyu-cs/MeanFlow
            - loss.py: SILoss, for the time sampling and the JVP
"""

import math

import torch
import torch.autograd.forward_ad as fwAD

from nexus_align.algorithms.base_algorithm import BaseAlgorithm


def jvp(fn, primals, tangents):
    """
    Calculate the Jacobian-vector product with forward-mode AD.

    torch.func.jvp does not accept the in-place unshard ops of FSDP, so this
    function uses the eager forward_ad API. It runs under no_grad because the
    JVP output is only a detached target. An unused autograd graph through an
    FSDP module breaks the state of its backward hooks.
    """
    with torch.no_grad(), fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        out, dudt = fwAD.unpack_dual(fn(*duals))
    return out, dudt


def sample_time_steps(
    batch_size, time_sampler, time_mu, time_sigma, ratio_r_not_equal_t, device
):
    """Sample time steps (r, t)."""
    # Sample r and t.
    if time_sampler == "uniform":
        time_samples = torch.rand(batch_size, 2, device=device)
    elif time_sampler == "logit_normal":
        time_samples = torch.randn(batch_size, 2, device=device)
        time_samples = torch.sigmoid(time_samples * time_sigma + time_mu)
    else:
        raise ValueError(f"❌ Unknown time sampler: {time_sampler}")
    sorted_samples, _ = torch.sort(time_samples, dim=1)  # make r < t
    r, t = sorted_samples[:, 0], sorted_samples[:, 1]

    # Set r = t for the first samples, as the official code does.
    num_equal = int(batch_size * (1.0 - ratio_r_not_equal_t))
    equal_mask = torch.arange(batch_size, device=device) < num_equal
    r = torch.where(equal_mask, t, r)

    return r, t


class MeanFlow(BaseAlgorithm):
    """MeanFlow training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Timestep configs.
        self.path_type = cfg_algorithm.get("path_type", "linear")  # timestep interpolant; "linear" or "cosine"
        self.time_sampler = cfg_algorithm.get("time_sampler", "logit_normal")  # timestep sampler; "uniform" or "logit_normal"
        self.time_mu = cfg_algorithm.get("time_mu", -0.4)  # mean of logit_normal, if time_sampler is logit_normal
        self.time_sigma = cfg_algorithm.get("time_sigma", 1.0)  # std of logit_normal, if time_sampler is logit_normal
        self.ratio_r_not_equal_t = cfg_algorithm.get("ratio_r_not_equal_t", 0.25)  # fraction of samples with r != t

        # CFG configs.
        self.cfg_prob = cfg_algorithm.get("cfg_prob", 0.1)  # probability to drop the class label for CFG
        self.cfg_omega = cfg_algorithm.get("cfg_omega", 0.2)  # CFG weight of the instantaneous velocity
        self.cfg_kappa = cfg_algorithm.get("cfg_kappa", 0.92)  # CFG weight of the mean velocity
        self.cfg_min_t = cfg_algorithm.get("cfg_min_t", 0.0)  # lowest timestep for CFG
        self.cfg_max_t = cfg_algorithm.get("cfg_max_t", 0.8)  # highest timestep for CFG

        # Loss weighting configs.
        self.weighting = cfg_algorithm.get("weighting", "adaptive")  # "uniform" (squared L2) or "adaptive"
        self.adaptive_p = cfg_algorithm.get("adaptive_p", 1.0)  # power p of the adaptive weight
        self.adaptive_eps = cfg_algorithm.get("adaptive_eps", 0.01)  # constant c added to the loss inside the adaptive weight

    def interpolant(self, t):
        """Return alpha_t, sigma_t, and their time derivatives at t."""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * math.pi / 2)
            sigma_t = torch.sin(t * math.pi / 2)
            d_alpha_t = -math.pi / 2 * torch.sin(t * math.pi / 2)
            d_sigma_t = math.pi / 2 * torch.cos(t * math.pi / 2)
        else:
            raise NotImplementedError(f"❌ Unknown time path type: {self.path_type}")

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def forward(self, data):
        """Calculate the MeanFlow loss of one batch."""
        # 1. Get data.
        z_0, labels = data
        z_0 = z_0.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        bs = z_0.shape[0]

        # 2. Drop the class labels of the first samples for CFG.
        unconditional_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)
        if self.cfg_prob > 0:
            labels = labels.clone()
            num_drop = (torch.rand(bs, device=self.device) < self.cfg_prob).sum()
            unconditional_mask = torch.arange(bs, device=self.device) < num_drop
            labels[unconditional_mask] = self.model.num_classes

        # 3. Sample timesteps r and t.
        r, t = sample_time_steps(
            bs, self.time_sampler, self.time_mu, self.time_sigma,
            self.ratio_r_not_equal_t, self.device,
        )
        time_diff = (t - r).view(-1, 1, 1, 1)

        # 4. Sample the noise, then make the noised latent and the velocity at t.
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        noises = torch.randn_like(z_0)
        z_t = alpha_t * z_0 + sigma_t * noises  # noised latent
        v_t = d_alpha_t * z_0 + d_sigma_t * noises  # conditional instantaneous velocity

        # 5. Predict the mean velocity from t to r.
        u = self.model(z_t, r, t, y=labels)

        # 6. Predict the conditional and unconditional velocities at t.
        null_labels = torch.full_like(labels, self.model.num_classes)
        with torch.no_grad():
            combined_u_at_t = self.model(
                torch.cat([z_t, z_t], dim=0),
                torch.cat([t, t], dim=0),
                torch.cat([t, t], dim=0),
                y=torch.cat([labels, null_labels], dim=0),
            )
        u_cond_at_t, u_uncond_at_t = torch.chunk(combined_u_at_t, 2, dim=0)

        # 7. Calculate the CFG velocity.
        v_tilde = (
            self.cfg_omega * v_t
            + self.cfg_kappa * u_cond_at_t
            + (1 - self.cfg_omega - self.cfg_kappa) * u_uncond_at_t
        )  # see paper Eq. 21
        cfg_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        v_hat = torch.where(cfg_mask.view(-1, 1, 1, 1), v_tilde, v_t)

        # 8. Calculate the JVP to get the target mean velocity.
        def fn_current(z, cur_r, cur_t):
            return self.model(z, cur_r, cur_t, y=labels)
        primals = (z_t, r, t)
        tangents = (v_hat, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(fn_current, primals, tangents)
        u_target = v_hat - time_diff * dudt  # see paper Eq. 10

        # 9. Calculate the loss from the prediction error.
        error = u - u_target.detach()
        error_sq = error.square()
        loss = error_sq.flatten(1).sum(dim=1)
        if self.weighting == "adaptive":
            loss = loss / (loss.detach() + self.adaptive_eps).pow(self.adaptive_p)

        loss = loss.mean()

        # Log the raw squared error for r = t (flow matching) and r != t (MeanFlow).
        loss_raw = error_sq.flatten(1).mean(dim=1).detach()
        r_ne_t = r != t

        return {
            "loss": loss,
            "metrics": {
                "loss": loss.detach(),
                "loss_raw": loss_raw.mean(),
                "loss_raw_r_eq_t": loss_raw[~r_ne_t].mean(),
                "loss_raw_r_ne_t": loss_raw[r_ne_t].mean(),
            },
        }

    @torch.no_grad()
    def sample(self, data):
        """
        Generate samples with z_r = z_t - (t - r) u(z_t, r, t).

        One step uses r = 0 and t = 1. More steps move t from 1 to 0 in num_steps.
        MeanFlow usually samples without CFG, so cfg_scale is 1.0.
        """
        # 1. Prepare the inputs.
        latents = data["latents"]
        y = data["labels"]
        cfg_scale = data.get("cfg_scale", 1.0)
        num_steps = data.get("num_steps", 1)
        batch_size = latents.shape[0]
        device = latents.device

        # 2. Set up CFG. We do not recommend CFG for MeanFlow.
        do_cfg = y is not None and cfg_scale > 1.0
        if do_cfg:
            null_y = torch.full_like(y, self.model.num_classes)

        # 3. Define the model call.
        def predict(z, r, t):
            """Predict the mean velocity, with CFG if do_cfg is true."""
            if not do_cfg:
                return self.model(z, r, t, y=y)
            u = self.model(
                torch.cat([z, z], dim=0),
                torch.cat([r, r], dim=0),
                torch.cat([t, t], dim=0),
                y=torch.cat([y, null_y], dim=0),
            )
            u_cond, u_uncond = u.chunk(2, dim=0)
            return u_uncond + cfg_scale * (u_cond - u_uncond)

        # 4. Denoise step by step.
        z = latents
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = torch.full((batch_size,), time_steps[i], device=device)
            r = torch.full((batch_size,), time_steps[i + 1], device=device)
            z = z - (t - r).view(-1, 1, 1, 1) * predict(z, r, t)

        return z
