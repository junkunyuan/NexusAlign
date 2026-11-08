"""
PixelDiT algorithm: flow matching on pixels with v-prediction and the REPA loss.

References:
    - PixelDiT:
        - Paper: PixelDiT: Pixel Diffusion Transformers for Image Generation
          (CVPR 2026)
        - Official code: https://github.com/NVlabs/PixelDiT
            - c2i/src/diffusion.py: BaseTrainer.preproprocess, REPATrainer,
              FlowDPMSolverSampler, time_shift_fn, simple_guidance_fn
            - c2i/src/lightning.py: LightningModel, for the bf16 autocast
"""

import torch
import torch.nn.functional as F

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.models.components.repa import DINOv2Encoder
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class PixelDiT(BaseAlgorithm):
    """
    PixelDiT training algorithm.

    t = 0 is the noise and t = 1 is the clean image, so z_t = t x + (1 - t) e. The
    model predicts the velocity v = x - e. The loss adds the REPA loss of the patch
    tokens; see paper Sec. 3.4.
    """

    def __init__(self, cfg_algorithm):
        super().__init__()

        # Timestep configs.
        self.time_mu = cfg_algorithm.get("time_mu", 0.0)  # logit_normal mean
        self.time_sigma = cfg_algorithm.get("time_sigma", 1.0)  # logit_normal std
        self.time_shift = cfg_algorithm.get("time_shift", 1.0)  # shift s of t to t / (t + s (1 - t)); s > 1 adds noise

        # CFG configs.
        self.cfg_prob = cfg_algorithm.get("cfg_prob", 0.1)  # probability to drop the class label for CFG

        # REPA configs.
        self.repa_weight = cfg_algorithm.get("repa_weight", 0.5)  # weight of the REPA loss
        self.repa_encoder_path = cfg_algorithm.get("repa_encoder_path", None)  # DINOv2 path of the REPA targets

        # Sampling configs.
        self.sample_time_shift = cfg_algorithm.get("sample_time_shift", 1.0)  # time shift of the sampling steps

        # Precision configs.
        self.autocast_dtype = cfg_algorithm.get("autocast_dtype", "bf16")  # model compute dtype

        # Training builds the REPA encoder at the first forward pass.
        self.repa_encoder = None

    @staticmethod
    def shift_time(t: torch.Tensor, shift: float) -> torch.Tensor:
        """Shift t to t / (t + shift (1 - t)), as the official time_shift_fn does."""
        return t / (t + (1 - t) * shift)

    def forward(self, data):
        """Calculate the PixelDiT loss of one batch."""
        # 1. Get data.
        x, labels = data
        x = x.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        bs = x.shape[0]

        # 2. Drop each class label with probability cfg_prob for CFG.
        drop = torch.rand(bs, device=self.device) < self.cfg_prob
        labels = torch.where(drop, self.model.num_classes, labels)

        # 3. Sample the timesteps, and shift them.
        t = torch.randn(bs, device=self.device) * self.time_sigma + self.time_mu
        t = self.shift_time(torch.sigmoid(t), self.time_shift)

        # 4. Sample the noise, then make the noised image and the velocity at t.
        t_ = t.view(-1, 1, 1, 1)
        noises = torch.randn_like(x)
        z_t = t_ * x + (1 - t_) * noises
        v_t = x - noises

        # 5. Calculate the flow matching loss and the REPA loss. As in the official
        # code, the losses also run under autocast, so the cosine runs in fp32.
        if self.repa_encoder is None:
            self.repa_encoder = DINOv2Encoder(self.repa_encoder_path).to(self.device)
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            v, z = self.model(z_t, t, labels, return_repa=True)
            loss_fm = (v - v_t).square().mean()
            target = self.repa_encoder(x, self.model.patch_size)
            loss_repa = (1 - F.cosine_similarity(z, target, dim=-1)).mean()
        loss = loss_fm + self.repa_weight * loss_repa

        return {
            "loss": loss,
            "metrics": {
                "loss": loss.detach(),
                "loss_fm": loss_fm.detach(),
                "loss_repa": loss_repa.detach(),
            },
        }

    @torch.no_grad()
    def sample(self, data):
        """
        Generate samples with FlowDPMSolver from t = 0 (noise) to t = 1 (image).

        The steps are linear in [0, 1 - 1 / num_steps], plus t = 1, and then shifted
        by sample_time_shift. The first step uses Euler, and the others use the
        two-step Adams-Bashforth update of the official FlowDPMSolverSampler. CFG
        applies only for cfg_min_t < t < cfg_max_t.
        """
        # 1. Prepare the inputs.
        z = data["latents"]
        y = data["labels"]
        cfg_scale = data.get("cfg_scale", 1.0)
        cfg_min_t = data.get("cfg_min_t", 0.0)
        cfg_max_t = data.get("cfg_max_t", 1.0)
        num_steps = data.get("num_steps", 100)
        batch_size = z.shape[0]
        null_y = torch.full_like(y, self.model.num_classes)

        # 2. Make the time steps on the CPU, as the official code does.
        time_steps = torch.linspace(0.0, 1 - 1.0 / num_steps, num_steps)
        time_steps = torch.cat([time_steps, torch.tensor([1.0])])
        time_steps = self.shift_time(time_steps, self.sample_time_shift).to(z.device)

        # 3. Solve the ODE. The official code puts the unconditional half first.
        v_prev = None
        for i in range(num_steps):
            t, t_next = time_steps[i], time_steps[i + 1]
            with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
                v = self.model(
                    torch.cat([z, z], dim=0),
                    t.repeat(2 * batch_size),
                    torch.cat([null_y, y], dim=0),
                )
            v_uncond, v_cond = v.chunk(2, dim=0)
            scale = cfg_scale if cfg_min_t < t < cfg_max_t else 1.0
            v = v_uncond + scale * (v_cond - v_uncond)
            dt = t_next - t
            z = z + v * dt if v_prev is None else z + dt * (1.5 * v - 0.5 * v_prev)
            v_prev = v

        return z
