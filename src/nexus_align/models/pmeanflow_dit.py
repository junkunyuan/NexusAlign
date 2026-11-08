"""
pMeanFlow DiT model: the iMeanFlow DiT on pixels, with x-prediction.

References:
    - pMeanFlow:
        - Paper: One-step Latent-free Image Generation with Pixel Mean Flows
          (arXiv 2026)
        - Official code: https://github.com/Lyy-iiis/pMF
            - models/pmfDiT.py: pmfDiT, precompute_rope_freqs_2d, apply_rotary_pos_emb
            - models/embedder.py: BottleneckPatchEmbedder
            - models/pmfDiT.py of the hsdp branch: the MLP width of pmfDiT_H
"""

import torch
import torch.nn as nn

from nexus_align.models.components.embedding import (
    BottleneckPatchEmbed, get_2d_rope_angles
)
from nexus_align.models.imeanflow_dit import IMeanFlowDiT, IMeanFlowDiTModel


class PMeanFlowDiT(IMeanFlowDiT):
    """
    Core architecture for PMeanFlowDiT.

    - The iMeanFlow DiT, with a bottleneck patch embedding of the pixels.
    - A learnable position embedding of all tokens, and a 2D RoPE of the patch
      tokens only.
    - Both heads predict a denoised image x. The model returns the velocities
      u = (z_t - x_u) / t and v = (z_t - x_v) / t; see paper Eq. 11.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 256,
        num_classes: int = 1000,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        bottleneck_dim: int = 128,
        mlp_multiple: int = 1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_multiple=mlp_multiple,
        )
        # Replace the patch embedding of iMeanFlow.
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.num_prefix_tokens + num_patches, hidden_size)
        )

        # 2D RoPE angles of the patches. The condition tokens have zero angles, so RoPE
        # does not change them.
        angles = get_2d_rope_angles(hidden_size // num_heads, self.x_embedder.grid_size)
        prefix = torch.zeros(self.num_prefix_tokens, angles.shape[1])
        angles = torch.cat([prefix, angles])
        self.rope_cos = angles.cos()  # [P + N, D / 2]
        self.rope_sin = angles.sin()  # [P + N, D / 2]
        self.initialize_pixel_weights()

    def initialize_pixel_weights(self) -> None:
        # Initialize both patch projections like nn.Linear (instead of nn.Conv2d):
        for proj in [self.x_embedder.proj1, self.x_embedder.proj2]:
            w = proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(proj.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)

    def embed(self, x, r, t, omega, t_min, t_max, y) -> torch.Tensor:
        return super().embed(x, r, t, omega, t_min, t_max, y) + self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        omega: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
        predict_u: bool = True,
        predict_v: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Predict the denoised images, and return them as velocities; see
        IMeanFlowDiT.forward for the arguments.

        The official code clips t to [0.05, 1] in the conversion.
        """
        x_u, x_v = super().forward(
            x, r, t, omega, t_min, t_max, y, predict_u=predict_u, predict_v=predict_v
        )
        t = t.clamp(0.05, 1.0).view(-1, 1, 1, 1)
        u = (x - x_u) / t if x_u is not None else None
        v = (x - x_v) / t if x_v is not None else None
        return u, v


# --------------------------------------------------------------------------------
# PMeanFlowDiT Configs.
# --------------------------------------------------------------------------------
PMeanFlowDiTModelNames = {
    "pmeanflowdit_h_32": dict(
        depth=48, hidden_size=1280, patch_size=32, num_heads=16, bottleneck_dim=256,
        mlp_multiple=8,
    ),
    "pmeanflowdit_l_32": dict(depth=32, hidden_size=1024, patch_size=32, num_heads=16),
    "pmeanflowdit_b_32": dict(depth=16, hidden_size=768, patch_size=32, num_heads=12),
    "pmeanflowdit_h_16": dict(
        depth=48, hidden_size=1280, patch_size=16, num_heads=16, bottleneck_dim=256,
        mlp_multiple=8,
    ),
    "pmeanflowdit_l_16": dict(depth=32, hidden_size=1024, patch_size=16, num_heads=16),
    "pmeanflowdit_b_16": dict(depth=16, hidden_size=768, patch_size=16, num_heads=12),
}


# --------------------------------------------------------------------------------
# PMeanFlowDiT Model Class.
# --------------------------------------------------------------------------------
class PMeanFlowDiTModel(IMeanFlowDiTModel):
    """
    FSDP-wrapped PMeanFlowDiT model.

    - Builds the configured PMeanFlowDiT variant.
    - Shards each IMeanFlowDiTBlock as an FSDP unit, as IMeanFlowDiTModel does.
    """

    def build_model(self) -> nn.Module:
        return PMeanFlowDiT(
            input_size=self.cfg_model.input_size,
            num_classes=self.cfg_model.num_classes,
            **PMeanFlowDiTModelNames[self.cfg_model.name],
        )
