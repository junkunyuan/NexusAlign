"""
DiT components: adaLN-Zero block, final layer, unpatchify, and 2D sin-cos positions.

References:
    - DiT:
        - Paper: Scalable Diffusion Models with Transformers
        - Official code: https://github.com/facebookresearch/DiT
"""

import numpy as np
import torch
import torch.nn as nn

from nexus_align.models.components.attention import Attention
from nexus_align.models.components.embedding import to_2tuple
from nexus_align.models.components.mlp import Mlp


# --------------------------------------------------------------------------------
# Transformer Blocks of DiT.
# --------------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT transformer block with adaLN-Zero conditioning.

    - Applies conditioned attention and MLP residual branches.
    - Uses the conditioning vector to produce shift, scale, and gate values.
    """
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        c: (B, D)
        return x: (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """
    Final DiT projection layer.

    - Applies conditioned normalization.
    - Projects hidden tokens back to patch pixels.
    """
    def __init__(
        self, hidden_size: int, patch_size: int | tuple[int, int], out_channels: int
    ) -> None:
        super().__init__()
        ph, pw = to_2tuple(patch_size)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


def unpatchify(
    x: torch.Tensor,
    patch_size: int | tuple[int, int],
    grid_size: tuple[int, int],
    out_channels: int,
) -> torch.Tensor:
    """
    x: (B, N, ph * pw * C), where N = gh * gw
    imgs: (B, C, gh * ph, gw * pw)
    """
    c = out_channels
    ph, pw = to_2tuple(patch_size)
    h, w = grid_size
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
    return imgs


# --------------------------------------------------------------------------------
# Sine/Cosine Positional Embedding Functions.
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int]
) -> np.ndarray:
    """
    grid_size: grid height and width, as an int or an (h, w) pair
    Returns pos_embed: (h*w, embed_dim).
    """
    h, w = to_2tuple(grid_size)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h.
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: positions to be encoded, size (M,)
    Returns an (M, D) array of embeddings.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
