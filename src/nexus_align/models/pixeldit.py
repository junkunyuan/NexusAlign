"""
PixelDiT model: a patch-level DiT for the semantics, and pixel-level PiT blocks for
the details.

References:
    - PixelDiT:
        - Paper: PixelDiT: Pixel Diffusion Transformers for Image Generation
          (CVPR 2026)
        - Official code: https://github.com/NVlabs/PixelDiT
            - pixdit_core/pixeldit_c2i.py: PixDiT, AugmentedDiTBlock,
              PixelTokenEmbedder, PiTBlock
            - pixdit_core/modules.py: RMSNorm, RotaryAttention, FeedForward, MLP,
              FinalLayer, precompute_freqs_cis_2d, apply_rotary_emb
            - c2i/src/diffusion.py: REPATrainer.proj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.dit import get_2d_sincos_pos_embed, unpatchify
from nexus_align.models.components.embedding import TimestepEmbedder, to_2tuple
from nexus_align.models.components.mlp import Mlp
from nexus_align.models.imeanflow_dit import SwiGLUMlp, apply_rotary_pos_emb


# --------------------------------------------------------------------------------
# Core PixelDiT Model.
# --------------------------------------------------------------------------------
class PixelDiT(nn.Module):
    """
    Core architecture for PixelDiT.

    - The patch-level pathway embeds the patches with a linear layer. Its DiT blocks
      use RoPE, and adaLN of the timestep t and the class label.
    - The pixel-level pathway embeds each pixel, and its PiT blocks refine the
      pixels. The condition of the PiT blocks is SiLU(s + t), where s is the output
      of the last DiT block; see paper Sec. 3.1.
    - Predicts the velocity in the shape of the input.
    - A REPA projector maps the tokens of DiT block repa_depth to repa_dim features.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 256,
        num_classes: int = 1000,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1152,
        depth: int = 26,
        num_heads: int = 16,
        pixel_hidden_size: int = 16,
        pixel_depth: int = 4,
        repa_depth: int = 8,
        repa_dim: int = 768,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        h, w = to_2tuple(input_size)
        self.grid_size = (h // patch_size, w // patch_size)
        self.repa_depth = repa_depth

        # Build the modules in the official order, so that the random initialization
        # of the same seed matches.
        self.pixel_embedder = PixelEmbed(
            input_size, patch_size, in_channels, pixel_hidden_size
        )
        self.s_embedder = nn.Linear(in_channels * patch_size ** 2, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=10)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 null class
        self.final_layer = FinalLayer(pixel_hidden_size, self.out_channels)
        self.patch_blocks = nn.ModuleList([
            PixelDiTBlock(hidden_size, num_heads, adaln_silu=False)
            for _ in range(depth)
        ])
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(pixel_hidden_size, hidden_size, patch_size, num_heads)
            for _ in range(pixel_depth)
        ])

        # RoPE angles of the patches. The PiT blocks attend over the patches too.
        angles = get_rope_angles(hidden_size // num_heads, self.grid_size)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)  # [N, D/2]
        self.register_buffer("rope_sin", angles.sin(), persistent=False)  # [N, D/2]
        self.initialize_weights()

        self.repa_proj = repa_projector(hidden_size, repa_dim)

    def initialize_weights(self) -> None:
        # Initialize the patch embedding like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.s_embedder.weight)
        nn.init.constant_(self.s_embedder.bias, 0)

        # Initialize the label embedding table:
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize the timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Set the output layer and the adaLN modulation layers to zero:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        for block in [*self.patch_blocks, *self.pixel_blocks]:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_repa: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W) tensor of noised images
        t: (B,) tensor of timesteps; t = 1 is the clean image
        y: (B,) tensor of class labels
        return_repa: if true, also return the projected REPA tokens
        return: (B, C, H, W) tensor of predicted velocities, and with return_repa the
            (B, N, repa_dim) tensor of REPA tokens
        """
        B = x.shape[0]
        rope = (self.rope_cos, self.rope_sin)
        t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        c = F.silu(t_emb + self.y_embedder(y).unsqueeze(1))  # (B, 1, D)

        # Patch-level pathway.
        patches = F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)
        s = self.s_embedder(patches)  # (B, N, D), where N = gh * gw
        for i, block in enumerate(self.patch_blocks):
            s = block(s, c, rope)
            if i + 1 == self.repa_depth:
                z = s
        s_cond = F.silu(t_emb + s).flatten(0, 1)  # (B * N, D)

        # Pixel-level pathway.
        x = self.pixel_embedder(x)  # (B * N, p * p, D_pix)
        for block in self.pixel_blocks:
            x = block(x, s_cond, rope)
        x = self.final_layer(x).reshape(B, s.shape[1], -1)  # (B, N, p * p * C)
        x = unpatchify(x, self.patch_size, self.grid_size, self.out_channels)
        if return_repa:
            return x, self.repa_proj(z)
        return x


# --------------------------------------------------------------------------------
# Transformer Blocks with RMSNorm, RoPE Attention, and adaLN.
# --------------------------------------------------------------------------------
def get_rope_angles(head_dim: int, grid_size: tuple[int, int]) -> torch.Tensor:
    """
    Return the (gh * gw, head_dim / 2) angles of the 2D RoPE of PixNerd.

    The positions span [0, 16] on both axes for any grid size. The channel pairs
    alternate between the column and the row positions.
    """
    gh, gw = grid_size
    freqs = torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim
    freqs = 1.0 / (10000 ** freqs)
    rows, cols = torch.meshgrid(
        torch.linspace(0, 16, gh), torch.linspace(0, 16, gw), indexing="ij"
    )
    angles = [torch.outer(pos.reshape(-1), freqs) for pos in (cols, rows)]
    return torch.stack(angles, dim=-1).reshape(gh * gw, -1)


def apply_adaln(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Modulate x with a shift and a scale of the same rank as x."""
    return x * (1 + scale) + shift


def repa_projector(hidden_size: int, repa_dim: int) -> nn.Sequential:
    """Return the REPA MLP from the hidden size to the REPA encoder features."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, repa_dim),
    )


class RMSNorm(nn.Module):
    """
    RMSNorm of PixNerd: it normalizes in fp32, then casts to the input dtype.

    The fp32 weight multiplies the result, so under autocast the output is fp32.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    """Multi-head self-attention with QK RMSNorm, RoPE, and no QKV biases."""
    def __init__(
        self, hidden_size: int, num_heads: int, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
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
        q = apply_rotary_pos_emb(self.q_norm(q), *rope).transpose(1, 2)
        k = apply_rotary_pos_emb(self.k_norm(k), *rope).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v.transpose(1, 2))  # (B, H, N, hd)
        x = self.proj(x.transpose(1, 2).reshape(B, N, D))
        return self.proj_drop(x)


class PixelDiTBlock(nn.Module):
    """
    Patch-level DiT block with RMSNorm, RoPE attention, a SwiGLU MLP, and adaLN.

    - c can have one vector per token, so adaLN can modulate each token differently.
    - If adaln_silu is true, adaLN applies SiLU to c before its linear layer.
    - The SwiGLU width rounds up to a multiple of mlp_multiple; 0 disables it.
    - proj_drop is the dropout after the attention output and inside the MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        mlp_multiple: int = 0,
        adaln_silu: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(2 * int(hidden_size * mlp_ratio) / 3)
        if mlp_multiple > 0:
            mlp_hidden_dim = -(-mlp_hidden_dim // mlp_multiple) * mlp_multiple
        self.mlp = SwiGLUMlp(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            *([nn.SiLU()] if adaln_silu else []),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, N, D)
        c: (B, 1, D) or (B, N, D)
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            apply_adaln(self.norm1(x), shift_msa, scale_msa), rope
        )
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


# --------------------------------------------------------------------------------
# Pixel-level Embedding, PiT Block, and Final Layer.
# --------------------------------------------------------------------------------
class PixelEmbed(nn.Module):
    """
    Embed each pixel with a linear layer, and add the fixed 2D sin-cos positions of
    the pixels. Then group the pixel tokens by patch.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int],
        patch_size: int,
        in_channels: int,
        pixel_hidden_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels, pixel_hidden_size)
        h, w = to_2tuple(input_size)
        pos_embed = get_2d_sincos_pos_embed(pixel_hidden_size, (h, w))
        pos_embed = torch.from_numpy(pos_embed).float().view(h, w, -1)
        self.register_buffer("pos_embed", pos_embed, persistent=False)  # [H, W, D_pix]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return x: (B * N, p * p, D_pix)
        """
        B, _, H, W = x.shape
        p = self.patch_size
        x = self.proj(x.permute(0, 2, 3, 1)) + self.pos_embed  # (B, H, W, D_pix)
        x = x.view(B, H // p, p, W // p, p, -1).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B * (H // p) * (W // p), p * p, -1)


class PiTBlock(nn.Module):
    """
    Pixel Transformer (PiT) block with pixel-wise adaLN and pixel token compaction.

    - Pixel-wise adaLN: each patch token of s_cond gives its own adaLN parameters to
      each pixel of the patch; see paper Eq. 7.
    - Pixel token compaction: the attention compacts the pixel tokens of a patch into
      one token, attends over the patches, and expands the output back to the pixels.
    """
    def __init__(
        self,
        pixel_hidden_size: int,
        hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        p2 = patch_size ** 2
        self.compress_to_attn = nn.Linear(p2 * pixel_hidden_size, hidden_size)
        self.expand_from_attn = nn.Linear(hidden_size, p2 * pixel_hidden_size)
        self.norm1 = RMSNorm(pixel_hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(pixel_hidden_size)
        self.mlp = Mlp(
            pixel_hidden_size, int(pixel_hidden_size * mlp_ratio), approximate="none"
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * p2 * pixel_hidden_size, bias=True)
        )

    def forward(
        self, x: torch.Tensor, s_cond: torch.Tensor, rope: tuple
    ) -> torch.Tensor:
        """
        x: (B * N, p * p, D_pix) pixel tokens
        s_cond: (B * N, D) patch tokens
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B * N, p * p, D_pix)
        """
        BN, P2, C = x.shape
        N = rope[0].shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(s_cond).view(BN, P2, -1).chunk(6, dim=-1)
        )
        x_attn = apply_adaln(self.norm1(x), shift_msa, scale_msa).view(BN, P2 * C)
        x_attn = self.compress_to_attn(x_attn).view(BN // N, N, -1)  # (B, N, D)
        x_attn = self.expand_from_attn(self.attn(x_attn, rope).view(BN, -1))
        x = x + gate_msa * x_attn.view(BN, P2, C)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final projection layer of the pixel tokens: RMSNorm, then a linear layer."""
    def __init__(self, pixel_hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(pixel_hidden_size)
        self.linear = nn.Linear(pixel_hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# --------------------------------------------------------------------------------
# PixelDiT Configs.
# --------------------------------------------------------------------------------
PixelDiTModelNames = {
    "pixeldit_xl_16": dict(
        depth=26, hidden_size=1152, patch_size=16, num_heads=16, pixel_depth=4
    ),
    "pixeldit_l_16": dict(
        depth=22, hidden_size=1024, patch_size=16, num_heads=16, pixel_depth=4
    ),
    "pixeldit_b_16": dict(
        depth=12, hidden_size=768, patch_size=16, num_heads=12, pixel_depth=2
    ),
}


# --------------------------------------------------------------------------------
# PixelDiT Model Class.
# --------------------------------------------------------------------------------
class PixelDiTModel(BaseModel):
    """
    FSDP-wrapped PixelDiT model.

    - Builds the configured PixelDiT variant.
    - Shards each PixelDiTBlock and each PiTBlock as an FSDP unit.
    - Inherits the EMA and FSDP setup from BaseModel.
    """

    def build_model(self) -> nn.Module:
        return PixelDiT(
            input_size=self.cfg_model.input_size,
            num_classes=self.cfg_model.num_classes,
            repa_dim=self.cfg_model.repa_dim,
            **PixelDiTModelNames[self.cfg_model.name],
        )

    def wrap_modules(self) -> tuple:
        return (PixelDiTBlock, PiTBlock)
