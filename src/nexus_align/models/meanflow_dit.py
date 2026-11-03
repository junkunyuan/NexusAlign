"""
MeanFlow DiT model: transformer-based diffusion backbone modified for MeanFlow.

References:
    - DiT:
        - Paper: Scalable Diffusion Models with Transformers (ICCV 2023)
        - Official code: https://github.com/facebookresearch/DiT
    - MeanFlow:
        - Paper: Mean Flows for One-step Generative Modeling (NeurIPS 2025)
        - Official code: https://github.com/Gsunshine/meanflow
            - models/models_dit.py: DiT
        - Unofficial code: https://github.com/zhuyu-cs/MeanFlow
            - sit.py: SiT.forward, for the embeddings of t and t - r
"""

import torch
import torch.nn as nn

from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.dit import (
    DiTBlock, FinalLayer, get_2d_sincos_pos_embed, unpatchify
)
from nexus_align.models.components.embedding import PatchEmbed, TimestepEmbedder


# --------------------------------------------------------------------------------
# Core MeanFlowDiT Model.
# --------------------------------------------------------------------------------
class MeanFlowDiT(nn.Module):
    """
    Core architecture for MeanFlowDiT.

    - Embeds the latent inputs as patches, and adds the fixed 2D sin-cos positions.
    - Conditions the blocks on the end timestep t, the interval length t - r, and the
      class labels.
    - Predicts the mean velocity in the shape of the input latent.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 32,
        num_classes: int = 1000,
        patch_size: int | tuple[int, int] = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)  # end timestep t
        self.r_embedder = TimestepEmbedder(hidden_size)  # interval length t - r
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 null class
        num_patches = self.x_embedder.num_patches
        # Use a fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize the transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize the frozen pos_embed with the sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize the label embedding table:
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize the timestep embedding MLPs:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

        # Set the adaLN modulation layers in the DiT blocks to zero:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Set the output layers to zero:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor of spatial inputs (images or latents)
        r: (B,) tensor of start timesteps
        t: (B,) tensor of end timesteps
        y: (B,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D), where N = gh * gw
        # Condition on the end timestep t and the interval length t - r.
        t_embed = self.t_embedder(t)  # (B, D)
        r_embed = self.r_embedder(t - r)  # (B, D)
        y_embed = self.y_embedder(y)  # (B, D)
        c = t_embed + r_embed + y_embed  # (B, D)

        for block in self.blocks:
            x = block(x, c)  # (B, N, D)
        x = self.final_layer(x, c)  # (B, N, ph * pw * out_channels)
        x = unpatchify(
            x, self.x_embedder.patch_size, self.x_embedder.grid_size, self.out_channels
        )  # (B, out_channels, H, W)

        return x


# --------------------------------------------------------------------------------
# MeanFlowDiT Configs.
# --------------------------------------------------------------------------------
MeanFlowDiTModelNames = {
    "meanflowdit_xl_2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    "meanflowdit_l_2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "meanflowdit_m_2": dict(depth=16, hidden_size=1024, patch_size=2, num_heads=16),
    "meanflowdit_b_2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
}


# --------------------------------------------------------------------------------
# MeanFlowDiT Model Class.
# --------------------------------------------------------------------------------
class MeanFlowDiTModel(BaseModel):
    """
    FSDP-wrapped MeanFlowDiT model.

    - Builds the configured MeanFlowDiT variant.
    - Shards each DiTBlock as an FSDP unit.
    - Inherits the EMA and FSDP setup from BaseModel.
    """

    def build_model(self) -> nn.Module:
        return MeanFlowDiT(
            input_size=self.cfg_model.input_size,
            num_classes=self.cfg_model.num_classes,
            **MeanFlowDiTModelNames[self.cfg_model.name],
        )

    def wrap_modules(self) -> tuple:
        return (DiTBlock,)
