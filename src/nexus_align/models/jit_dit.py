"""
JiT DiT model: a plain DiT on large pixel patches that predicts the clean image.

References:
    - JiT:
        - Paper: Back to Basics: Let Denoising Generative Models Denoise (arXiv 2025)
        - Official code: https://github.com/LTH14/JiT
            - model_jit.py: JiT, JiTBlock, Attention, scaled_dot_product_attention,
              SwiGLUFFN, FinalLayer, and the JiT_{B,L,H}_{16,32} configs
            - util/model_util.py: RMSNorm, VisionRotaryEmbeddingFast with
              num_cls_token
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.dit import (
    get_2d_sincos_pos_embed, modulate, unpatchify
)
from nexus_align.models.components.embedding import (
    BottleneckPatchEmbed, TimestepEmbedder, get_2d_rope_angles, to_2tuple
)
from nexus_align.models.imeanflow_dit import apply_rotary_pos_emb


# --------------------------------------------------------------------------------
# Core JiTDiT Model.
# --------------------------------------------------------------------------------
class JiTDiT(nn.Module):
    """
    Core architecture for JiTDiT.

    - Embeds the pixel patches through a bottleneck, and adds the fixed 2D sin-cos
      positions. A 2D RoPE also gives the positions of the patch tokens.
    - Conditions the blocks on the timestep t and the class label with adaLN-Zero.
    - Puts in_context_len class tokens before the patch tokens at block
      in_context_start. Each token is the label embedding plus a learnable position.
    - Predicts the clean image in the shape of the input.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 256,
        num_classes: int = 1000,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 4,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_classes = num_classes
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 null class
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size,
            proj1_bias=False,
        )
        num_patches = self.x_embedder.num_patches
        # Use a fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.in_context_posemb = nn.Parameter(
            torch.empty(1, in_context_len, hidden_size)
        )
        nn.init.normal_(self.in_context_posemb, std=0.02)

        # 2D RoPE angles of the patches. The class tokens have zero angles, so RoPE
        # does not change them.
        angles = get_2d_rope_angles(hidden_size // num_heads, self.x_embedder.grid_size)
        angles = torch.cat([torch.zeros(in_context_len, angles.shape[1]), angles])
        self.register_buffer("rope_cos", angles.cos(), persistent=False)  # [P + N, D/2]
        self.register_buffer("rope_sin", angles.sin(), persistent=False)  # [P + N, D/2]

        # Use dropout only in the middle half of the blocks (paper Appendix A).
        self.blocks = nn.ModuleList([
            JiTDiTBlock(
                hidden_size, num_heads, mlp_ratio,
                proj_drop=proj_drop if depth // 4 <= i < depth // 4 * 3 else 0.0,
            )
            for i in range(depth)
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

        # Initialize both patch projections like nn.Linear (instead of nn.Conv2d):
        for proj in [self.x_embedder.proj1, self.x_embedder.proj2]:
            w = proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize the label embedding table:
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize the timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Set the adaLN modulation layers in the blocks to zero:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Set the output layers to zero:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor of noised images
        t: (B,) tensor of timesteps; t = 1 is the clean image
        y: (B,) tensor of class labels
        return: (B, C, H, W) tensor of predicted clean images
        """
        y_embed = self.y_embedder(y)  # (B, D)
        c = self.t_embedder(t) + y_embed  # (B, D)
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D), where N = gh * gw

        P = self.in_context_len
        for i, block in enumerate(self.blocks):
            if i == self.in_context_start:
                tokens = y_embed.unsqueeze(1) + self.in_context_posemb  # (B, P, D)
                x = torch.cat([tokens, x], dim=1)  # (B, P + N, D)
            start = 0 if i >= self.in_context_start else P
            x = block(x, c, (self.rope_cos[start:], self.rope_sin[start:]))

        x = self.final_layer(x[:, P:], c)  # (B, N, ph * pw * out_channels)
        return unpatchify(
            x, self.x_embedder.patch_size, self.x_embedder.grid_size, self.out_channels
        )  # (B, out_channels, H, W)


# --------------------------------------------------------------------------------
# Transformer Blocks with adaLN-Zero, RoPE Attention, and SwiGLU.
# --------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm that normalizes in fp32 and returns the input dtype, as LLaMA does."""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class Attention(nn.Module):
    """
    Multi-head self-attention with QK RMSNorm and RoPE.

    As in the official code, the RoPE outputs and the attention logits stay in fp32,
    also under autocast.
    """
    def __init__(
        self, hidden_size: int, num_heads: int, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, N, D)
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B, N, D)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, N, H, head_dim) each
        q = apply_rotary_pos_emb(self.q_norm(q).float(), *rope).transpose(1, 2)
        k = apply_rotary_pos_emb(self.k_norm(k).float(), *rope).transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            attn = q @ k.transpose(-2, -1) * (1 / math.sqrt(self.head_dim))
        x = attn.softmax(dim=-1) @ v.transpose(1, 2)  # (B, H, N, head_dim)
        x = self.proj(x.transpose(1, 2).reshape(B, N, D))
        return self.proj_drop(x)


class SwiGLUMlp(nn.Module):
    """MLP with a Swish-gated linear unit (SwiGLU) of width 2/3 of mlp_hidden_dim."""
    def __init__(
        self, hidden_size: int, mlp_hidden_dim: int, drop: float = 0.0
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(mlp_hidden_dim * 2 / 3)
        self.w12 = nn.Linear(hidden_size, 2 * mlp_hidden_dim)
        self.w3 = nn.Linear(mlp_hidden_dim, hidden_size)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class JiTDiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero, RMSNorm, RoPE attention, and a SwiGLU MLP.

    proj_drop is the dropout after the attention output and inside the MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, N, D)
        c: (B, D)
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final projection layer: RMSNorm with adaLN, then a linear layer."""
    def __init__(
        self, hidden_size: int, patch_size: int | tuple[int, int], out_channels: int
    ) -> None:
        super().__init__()
        ph, pw = to_2tuple(patch_size)
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, ph * pw * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        c: (B, D)
        return x: (B, N, ph * pw * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


# --------------------------------------------------------------------------------
# JiTDiT Configs.
# --------------------------------------------------------------------------------
JiTDiTModelNames = {
    "jitdit_h_32": dict(
        depth=32, hidden_size=1280, patch_size=32, num_heads=16, bottleneck_dim=256,
        in_context_start=10, proj_drop=0.2,
    ),
    "jitdit_l_32": dict(
        depth=24, hidden_size=1024, patch_size=32, num_heads=16, in_context_start=8
    ),
    "jitdit_b_32": dict(
        depth=12, hidden_size=768, patch_size=32, num_heads=12, in_context_start=4
    ),
    "jitdit_h_16": dict(
        depth=32, hidden_size=1280, patch_size=16, num_heads=16, bottleneck_dim=256,
        in_context_start=10, proj_drop=0.2,
    ),
    "jitdit_l_16": dict(
        depth=24, hidden_size=1024, patch_size=16, num_heads=16, in_context_start=8
    ),
    "jitdit_b_16": dict(
        depth=12, hidden_size=768, patch_size=16, num_heads=12, in_context_start=4
    ),
}


# --------------------------------------------------------------------------------
# JiTDiT Model Class.
# --------------------------------------------------------------------------------
class JiTDiTModel(BaseModel):
    """
    FSDP-wrapped JiTDiT model.

    - Builds the configured JiTDiT variant.
    - Shards each JiTDiTBlock as an FSDP unit.
    - Inherits the EMA and FSDP setup from BaseModel.
    """

    def build_model(self) -> nn.Module:
        return JiTDiT(
            input_size=self.cfg_model.input_size,
            num_classes=self.cfg_model.num_classes,
            **JiTDiTModelNames[self.cfg_model.name],
        )

    def wrap_modules(self) -> tuple:
        return (JiTDiTBlock,)
