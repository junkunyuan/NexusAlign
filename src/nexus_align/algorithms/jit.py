"""
JiT algorithm: flow matching on pixels with x-prediction and the v-loss.

References:
    - JiT:
        - Paper: Back to Basics: Let Denoising Generative Models Denoise (arXiv 2025)
        - Official code: https://github.com/LTH14/JiT
            - denoiser.py: Denoiser.drop_labels, sample_t, forward, generate,
              _forward_sample, _euler_step, _heun_step
            - engine_jit.py: train_one_epoch and evaluate, for the bf16 autocast
"""

import torch

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class JiT(BaseAlgorithm):
    """
    JiT training algorithm.

    As in the paper, t = 0 is the noise and t = 1 is the clean image, so
    z_t = t x + (1 - t) e. The model predicts x, and the loss compares the velocities
    v = (x - z_t) / (1 - t).
    """

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Timestep configs.
        self.time_mu = cfg_algorithm.get("time_mu", -0.8)  # logit_normal mean; see paper Table 3
        self.time_sigma = cfg_algorithm.get("time_sigma", 0.8)  # logit_normal std
        self.t_eps = cfg_algorithm.get("t_eps", 0.05)  # lower clip of 1 - t in the velocity division

        # Noise configs.
        self.noise_scale = cfg_algorithm.get("noise_scale", 1.0)  # noise std; image_size / 256 in paper Table 9

        # CFG configs.
        self.cfg_prob = cfg_algorithm.get("cfg_prob", 0.1)  # probability to drop the class label for CFG

        # Precision configs.
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "bf16")  # model compute dtype

    def velocity(self, z_t, t, labels):
        """Predict x, and return the velocity (x - z_t) / (1 - t); see paper Eq. 6."""
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            x = self.model(z_t, t, labels)
        return (x - z_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

    def forward(self, data):
        """Calculate the JiT loss of one batch."""
        # 1. Get data.
        x, labels = data
        x = x.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        bs = x.shape[0]

        # 2. Drop each class label with probability cfg_prob for CFG.
        drop = torch.rand(bs, device=self.device) < self.cfg_prob
        labels = torch.where(drop, self.model.num_classes, labels)

        # 3. Sample the timesteps.
        t = torch.randn(bs, device=self.device) * self.time_sigma + self.time_mu
        t = torch.sigmoid(t)

        # 4. Sample the noise, then make the noised image and the velocity at t.
        t_ = t.view(-1, 1, 1, 1)
        noises = torch.randn_like(x) * self.noise_scale
        z_t = t_ * x + (1 - t_) * noises  # see paper Eq. 1
        v_t = (x - z_t) / (1 - t_).clamp_min(self.t_eps)

        # 5. Calculate the v-loss of the predicted clean image.
        v = self.velocity(z_t, t, labels)
        loss = (v_t - v).square().mean(dim=(1, 2, 3)).mean()

        return {"loss": loss, "metrics": {"loss": loss.detach()}}

    @torch.no_grad()
    def sample(self, data):
        """
        Generate samples by solving the ODE from t = 0 (noise) to t = 1 (image).

        The steps are linear in [0, 1]. All steps but the last use Heun, and the last
        uses Euler. CFG applies only for cfg_min_t < t < cfg_max_t, and for all
        t < cfg_max_t if cfg_min_t is 0.
        """
        # 1. Prepare the inputs.
        z = data["latents"] * self.noise_scale
        y = data["labels"]
        cfg_scale = data.get("cfg_scale", 1.0)
        cfg_min_t = data.get("cfg_min_t", 0.0)
        cfg_max_t = data.get("cfg_max_t", 1.0)
        num_steps = data.get("num_steps", 50)
        batch_size = z.shape[0]
        null_y = torch.full_like(y, self.model.num_classes)

        # 2. Define the CFG velocity at a step.
        def guided_velocity(z, t):
            v = self.velocity(
                torch.cat([z, z], dim=0),
                t.repeat(2 * batch_size),
                torch.cat([y, null_y], dim=0),
            )
            v_cond, v_uncond = v.chunk(2, dim=0)
            # Compare t in fp32, as the official code does.
            in_interval = (t < cfg_max_t) & ((cfg_min_t == 0) | (t > cfg_min_t))
            scale = torch.where(in_interval, cfg_scale, 1.0)
            return v_uncond + scale * (v_cond - v_uncond)

        # 3. Solve the ODE.
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=z.device)
        for i in range(num_steps):
            t, t_next = time_steps[i], time_steps[i + 1]
            v = guided_velocity(z, t)
            if i < num_steps - 1:
                v_next = guided_velocity(z + (t_next - t) * v, t_next)
                v = 0.5 * (v + v_next)
            z = z + (t_next - t) * v

        return z
