"""
iMeanFlow algorithm: a guidance-conditioned v-loss that uses the mean velocity to predict v.

References:
    - iMeanFlow:
        - Paper: Improved Mean Flows: On the Challenges of Fastforward Generative Models
          (CVPR 2026)
        - Official code: https://github.com/Lyy-iiis/imeanflow
            - imf.py: iMeanFlow.forward, sample_tr, sample_cfg_scale,
              sample_cfg_interval, cond_drop, guidance_fn, generate
"""

import math

import torch

from nexus_align.algorithms.meanflow import jvp, sample_time_steps
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class IMeanFlow(BaseAlgorithm):
    """iMeanFlow training algorithm."""

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Timestep configs.
        self.time_mu = cfg_algorithm.get("time_mu", -0.4)  # logit_normal mean
        self.time_sigma = cfg_algorithm.get("time_sigma", 1.0)  # logit_normal std
        self.ratio_r_not_equal_t = cfg_algorithm.get("ratio_r_not_equal_t", 0.5)  # fraction of samples with r != t

        # CFG configs.
        self.cfg_prob = cfg_algorithm.get("cfg_prob", 0.1)  # probability to drop the class label for CFG
        self.cfg_omega_max = cfg_algorithm.get("cfg_omega_max", 8.0)  # sample omega from [1, cfg_omega_max]
        self.cfg_beta = cfg_algorithm.get("cfg_beta", 1.0)  # omega ~ p(omega) ∝ omega^-beta
        self.cfg_min_t_range = cfg_algorithm.get("cfg_min_t_range", [0.0, 0.5])  # lowest timestep for CFG; t_min ~ U(range)
        self.cfg_max_t_range = cfg_algorithm.get("cfg_max_t_range", [0.5, 1.0])  # highest timestep for CFG; t_max ~ U(range)

        # Loss weighting configs.
        self.adaptive_p = cfg_algorithm.get("adaptive_p", 1.0)  # power p of the adaptive weight
        self.adaptive_eps = cfg_algorithm.get("adaptive_eps", 0.01)  # constant c added to the loss inside the adaptive weight

    def sample_cfg_scale(self, batch_size):
        """Sample guidance scales omega ~ p(omega) ∝ omega^-beta on [1, omega_max]."""
        u = torch.rand(batch_size, device=self.device)
        if self.cfg_beta == 1.0:
            return torch.exp(u * math.log(self.cfg_omega_max))
        power = 1.0 - self.cfg_beta
        return (1 + u * (self.cfg_omega_max ** power - 1)) ** (1 / power)

    def sample_cfg_interval(self, batch_size, full_mask):
        """Sample guidance intervals [t_min, t_max]. Masked samples get [0, 1]."""
        def uniform(low, high):
            return low + (high - low) * torch.rand(batch_size, device=self.device)
        t_min = torch.where(full_mask, 0.0, uniform(*self.cfg_min_t_range))
        t_max = torch.where(full_mask, 1.0, uniform(*self.cfg_max_t_range))
        return t_min, t_max

    def sample_r_t(self, batch_size):
        """Sample the time steps (r, t), with r = t for the first samples."""
        return sample_time_steps(
            batch_size, "logit_normal", self.time_mu, self.time_sigma,
            self.ratio_r_not_equal_t, self.device,
        )

    def add_noise(self, z_0, t):
        """Return the noised latent z_t and the conditional velocity at t."""
        t = t.view(-1, 1, 1, 1)
        noises = torch.randn_like(z_0)
        z_t = (1 - t) * z_0 + t * noises
        return z_t, noises - z_0

    def aux_loss(self, z_0, z_t, t, u):
        """Return the auxiliary loss of each sample and its metrics. iMeanFlow has none."""
        return torch.zeros_like(t), {}

    def adaptive_loss(self, error):
        """Weight the squared error of each sample adaptively; see MeanFlow Eq. 22."""
        loss = error.square().flatten(1).sum(dim=1)
        return self.adaptive_weight(loss)

    def adaptive_weight(self, loss):
        """Divide the loss of each sample by (loss + c)^p, without gradient."""
        return loss / (loss.detach() + self.adaptive_eps).pow(self.adaptive_p)

    def forward(self, data):
        """Calculate the iMeanFlow loss of one batch."""
        # 1. Get data.
        z_0, labels = data
        z_0 = z_0.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        bs = z_0.shape[0]

        # 2. Sample timesteps r and t.
        r, t = self.sample_r_t(bs)
        time_diff = (t - r).view(-1, 1, 1, 1)

        # 3. Sample the guidance scale and the guidance interval.
        omega = self.sample_cfg_scale(bs)
        cfg_min_t, cfg_max_t = self.sample_cfg_interval(bs, r == t)

        # 4. Sample the noise, then make the noised latent and the velocity at t.
        z_t, v_t = self.add_noise(z_0, t)

        # 5. Drop the class labels of the first samples for CFG.
        null_labels = torch.full_like(labels, self.model.num_classes)
        num_drop = (torch.rand(bs, device=self.device) < self.cfg_prob).sum()
        unconditional_mask = torch.arange(bs, device=self.device) < num_drop
        dropped_labels = torch.where(unconditional_mask, null_labels, labels)  # see paper Eq. 16

        # 6. Predict the mean velocity from t to r and the instantaneous velocity at t.
        u, v = self.model(z_t, r, t, omega, cfg_min_t, cfg_max_t, y=dropped_labels)

        # 7. Predict the conditional and unconditional velocities at t with the v head.
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        in_interval = (t >= cfg_min_t) & (t <= cfg_max_t)
        omega_at_t = torch.where(in_interval, omega, ones)
        with torch.no_grad():
            _, combined_v_at_t = self.model(
                torch.cat([z_t, z_t], dim=0),
                torch.cat([t, t], dim=0),
                torch.cat([t, t], dim=0),
                torch.cat([omega_at_t, ones], dim=0),
                torch.cat([zeros, zeros], dim=0),
                torch.cat([ones, ones], dim=0),
                y=torch.cat([labels, null_labels], dim=0),
                predict_u=False,
            )
        v_cond_at_t, v_uncond_at_t = torch.chunk(combined_v_at_t, 2, dim=0)

        # 8. Calculate the CFG velocity target; see paper Eq. 17.
        scale = (1 - 1 / omega_at_t).view(-1, 1, 1, 1)
        v_target = v_t + scale * (v_cond_at_t - v_uncond_at_t)
        v_target = torch.where(unconditional_mask.view(-1, 1, 1, 1), v_t, v_target)

        # 9. Calculate the compound prediction V = u + (t - r) sg(du/dt).
        def fn_current(z, cur_r, cur_t):
            return self.model(
                z, cur_r, cur_t, omega, cfg_min_t, cfg_max_t,
                y=dropped_labels, predict_v=False,
            )[0]
        primals = (z_t, r, t)
        tangents = (v_cond_at_t, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(fn_current, primals, tangents)
        V = u + time_diff * dudt  # see paper Eq. 15

        # 10. Calculate the loss from the errors of the u head and the auxiliary v head.
        error_u = V - v_target
        error_v = v - v_target
        loss_u = self.adaptive_loss(error_u)
        loss_v = self.adaptive_loss(error_v)
        loss_aux, aux_metrics = self.aux_loss(z_0, z_t, t, u)
        loss = (loss_u + loss_v + loss_aux).mean()

        # Log the raw squared errors of the u head and the auxiliary v head for r = t
        # (flow matching) and r != t (MeanFlow). Only training uses the v head.
        loss_raw_u = error_u.square().flatten(1).mean(dim=1).detach()
        loss_raw_v = error_v.square().flatten(1).mean(dim=1).detach()
        r_ne_t = r != t

        return {
            "loss": loss,
            "metrics": {
                "loss": loss.detach(),
                "loss_raw_u": loss_raw_u.mean(),
                "loss_raw_u_r_eq_t": loss_raw_u[~r_ne_t].mean(),
                "loss_raw_u_r_ne_t": loss_raw_u[r_ne_t].mean(),
                "loss_raw_v": loss_raw_v.mean(),
                "loss_raw_v_r_eq_t": loss_raw_v[~r_ne_t].mean(),
                "loss_raw_v_r_ne_t": loss_raw_v[r_ne_t].mean(),
                **aux_metrics,
            },
        }

    @torch.no_grad()
    def sample(self, data):
        """
        Generate samples with z_r = z_t - (t - r) u(z_t, r, t | y, omega, interval).

        One step uses r = 0 and t = 1. More steps move t from 1 to 0 in num_steps.
        Guidance is a model condition, so each step needs only one forward pass.
        """
        latents = data["latents"]
        y = data["labels"]
        cfg_scale = data.get("cfg_scale", 1.0)
        cfg_min_t = data.get("cfg_min_t", 0.0)
        cfg_max_t = data.get("cfg_max_t", 1.0)
        num_steps = data.get("num_steps", 1)

        batch_size = latents.shape[0]
        device = latents.device

        omega = torch.full((batch_size,), cfg_scale, device=device)
        t_min = torch.full((batch_size,), cfg_min_t, device=device)
        t_max = torch.full((batch_size,), cfg_max_t, device=device)

        z = latents
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur, t_next = time_steps[i], time_steps[i + 1]
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)
            u, _ = self.model(z, r, t, omega, t_min, t_max, y=y, predict_v=False)
            z = z - (t_cur - t_next) * u

        return z
