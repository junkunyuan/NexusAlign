"""
PixelDiT2 algorithm: JiT with representation grounding and the REPA loss.

References:
    - PixelDiT2:
        - Paper: PixelDiT2: Representation-Grounded Pixel Diffusion Transformers
          (NeurIPS 2026)
        - Official code: https://github.com/NVlabs/PixelDiT
            - c2i/src/grounded_diffusion.py: GroundedFlowTrainer, GroundedHeunSampler
            - c2i/src/lightning.py: LightningModel, for the bf16 autocast
"""

import torch
import torch.nn.functional as F

from nexus_align.utils.dtype import DTYPE_MAP
from nexus_align.models.components.repa import DINOv2Encoder
from nexus_align.algorithms.jit import JiT


class PixelDiT2(JiT):
    """
    PixelDiT2 training algorithm.

    It is JiT with two additions: the model grounds its condition on a frozen DINOv3,
    and the loss adds the REPA loss of the patch tokens. Each sample also drops its
    grounding tokens with grounding_drop_prob, so that CFG has a matching
    unconditional branch; see paper Sec. 3.2. Sampling is the JiT sampling, and the
    model drops the grounding tokens of the null class.
    """

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)

        # Grounding dropout configs.
        self.grounding_drop_prob = cfg_algorithm.get("grounding_drop_prob", 0.1)  # probability to drop the grounding tokens for CFG
        self.joint_drop_prob = cfg_algorithm.get("joint_drop_prob", 0.0)  # probability to drop both the label and the grounding tokens

        # REPA configs.
        self.repa_weight = cfg_algorithm.get("repa_weight", 0.5)  # weight of the REPA loss
        self.repa_encoder_path = cfg_algorithm.get("repa_encoder_path", None)  # DINOv2 path of the REPA targets

        # Training builds the REPA encoder at the first forward pass.
        self.repa_encoder = None

    def forward(self, data):
        """Calculate the PixelDiT2 loss of one batch."""
        # 1. Get data.
        x, labels = data
        x = x.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        bs = x.shape[0]

        # 2. Drop the class labels and the grounding tokens for CFG, in the order of
        # the official random draws.
        drop = torch.rand(bs, device=self.device) < self.cfg_prob
        drop_grounding = torch.rand(bs, device=self.device) < self.grounding_drop_prob
        if self.joint_drop_prob > 0:
            joint = torch.rand(bs, device=self.device) < self.joint_drop_prob
            drop, drop_grounding = drop | joint, drop_grounding | joint
        labels = torch.where(drop, self.model.num_classes, labels)

        # 3. Sample the timesteps.
        t = torch.randn(bs, device=self.device) * self.time_sigma + self.time_mu
        t = torch.sigmoid(t)

        # 4. Sample the noise, then make the noised image and the velocity at t.
        t_ = t.view(-1, 1, 1, 1)
        noises = torch.randn_like(x) * self.noise_scale
        z_t = t_ * x + (1 - t_) * noises  # see paper Eq. 1
        v_t = (x - z_t) / (1 - t_).clamp_min(self.t_eps)

        # 5. Calculate the v-loss of the predicted clean image and the REPA loss. As
        # in the official code, the losses also run under autocast, so the cosine
        # runs in fp32.
        if self.repa_encoder is None:
            self.repa_encoder = DINOv2Encoder(self.repa_encoder_path).to(self.device)
        with torch.autocast("cuda", dtype=DTYPE_MAP[self.autocast_dtype]):
            x_pred, z = self.model(z_t, t, labels, drop_grounding, return_repa=True)
            v = (x_pred - z_t) / (1 - t_).clamp_min(self.t_eps)  # see paper Eq. 2
            loss_fm = (v_t - v).square().mean(dim=(1, 2, 3)).mean()
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
