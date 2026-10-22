"""
iMeanFlow DiT model: in-context conditioned transformer with u and v heads.

References:
    - iMeanFlow:
        - Paper: Improved Mean Flows: On the Challenges of Fastforward Generative Models
        - Official code: https://github.com/Lyy-iiis/imeanflow
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.dit import unpatchify
from nexus_align.models.components.embedding import (
    PatchEmbed, TimestepEmbedder, to_2tuple
)


# --------------------------------------------------------------------------------
# Core IMeanFlowDiT Model.
# --------------------------------------------------------------------------------
class IMeanFlowDiT(nn.Module):
    """
    Core architecture for IMeanFlowDiT.

    - Patch-embeds latent inputs; positions come from 1D RoPE over the full sequence.
    - Conditions in context: class label, guidance scale, guidance interval, and
      interval length t - r are each replicated into learnable tokens that are
      prepended to the patch tokens; there is no adaLN.
    - A shared trunk feeds two heads: the u head predicts the mean velocity, the
      auxiliary v head predicts the instantaneous velocity (training only).
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 32,
        num_classes: int = 1000,
        patch_size: int | tuple[int, int] = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        weight_init_constant: float = 0.32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.weight_init_constant = weight_init_constant

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)  # interval length t - r
        self.omega_embedder = TimestepEmbedder(hidden_size)  # scale as 1 - 1/omega
        self.t_min_embedder = TimestepEmbedder(hidden_size)  # guidance interval start
        self.t_max_embedder = TimestepEmbedder(hidden_size)  # guidance interval end
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 uncond row

        # Learnable tokens per condition; a condition embedding is added to its tokens.
        self.class_tokens = nn.Parameter(torch.empty(num_class_tokens, hidden_size))
        self.omega_tokens = nn.Parameter(torch.empty(num_cfg_tokens, hidden_size))
        self.t_min_tokens = nn.Parameter(torch.empty(num_interval_tokens, hidden_size))
        self.t_max_tokens = nn.Parameter(torch.empty(num_interval_tokens, hidden_size))
        self.time_tokens = nn.Parameter(torch.empty(num_time_tokens, hidden_size))
        self.num_prefix_tokens = (
            num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens
            + num_time_tokens
        )

        # RoPE angles over the full sequence, conditioning tokens first.
        head_dim = hidden_size // num_heads
        seq_len = self.num_prefix_tokens + self.x_embedder.num_patches
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        freqs = 1.0 / (10000 ** freqs)
        angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)  # [N, D / 2]
        self.register_buffer("rope_sin", angles.sin(), persistent=False)  # [N, D / 2]

        def blocks(n):
            return nn.ModuleList([
                IMeanFlowDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(n)
            ])
        self.shared_blocks = blocks(depth - head_depth)
        self.u_blocks = blocks(head_depth)
        self.v_blocks = blocks(head_depth)
        self.u_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.v_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize linear layers as N(0, c^2 / fan_in) with zero bias; c = 1 for the
        # embedders and 0.32 inside the transformer blocks (paper Appendix A).
        def _scaled_init(module, constant):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    std = constant / math.sqrt(layer.in_features)
                    nn.init.normal_(layer.weight, std=std)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        for embedder in [self.h_embedder, self.omega_embedder,
                         self.t_min_embedder, self.t_max_embedder]:
            _scaled_init(embedder, 1.0)
        for block in [*self.shared_blocks, *self.u_blocks, *self.v_blocks]:
            _scaled_init(block, self.weight_init_constant)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table and condition tokens:
        hidden_size = self.y_embedder.embedding_dim
        nn.init.normal_(self.y_embedder.weight, std=1 / math.sqrt(hidden_size))
        for tokens in [self.class_tokens, self.omega_tokens, self.t_min_tokens,
                       self.t_max_tokens, self.time_tokens]:
            nn.init.normal_(tokens, std=1 / math.sqrt(hidden_size))

        # Zero-out output layers:
        for final_layer in [self.u_final_layer, self.v_final_layer]:
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

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
        x: (B, C, H, W) tensor of spatial inputs (images or latents)
        r: (B,) tensor of start timesteps
        t: (B,) tensor of end timesteps
        omega: (B,) tensor of guidance scales
        t_min: (B,) tensor of guidance interval starts
        t_max: (B,) tensor of guidance interval ends
        y: (B,) tensor of class labels
        predict_u, predict_v: which heads to run; a skipped head returns None
        return (u, v): mean velocity over [r, t] and instantaneous velocity at t
        """
        x = self.x_embedder(x)  # (B, N, D), where N = gh * gw
        # Condition on interval length t - r only, not on t (official code).
        h_embed = self.h_embedder(t - r)  # (B, D)
        omega_embed = self.omega_embedder(1 - 1 / omega)  # (B, D)
        t_min_embed = self.t_min_embedder(t_min)  # (B, D)
        t_max_embed = self.t_max_embedder(t_max)  # (B, D)
        y_embed = self.y_embedder(y)  # (B, D)
        x = torch.cat([
            self.class_tokens + y_embed.unsqueeze(1),
            self.omega_tokens + omega_embed.unsqueeze(1),
            self.t_min_tokens + t_min_embed.unsqueeze(1),
            self.t_max_tokens + t_max_embed.unsqueeze(1),
            self.time_tokens + h_embed.unsqueeze(1),
            x,
        ], dim=1)  # (B, P + N, D)

        rope = (self.rope_cos, self.rope_sin)
        for block in self.shared_blocks:
            x = block(x, rope)

        def run_head(blocks, final_layer):
            tokens = x
            for block in blocks:
                tokens = block(tokens, rope)
            tokens = final_layer(tokens[:, self.num_prefix_tokens:])  # (B, N, ph*pw*C)
            return unpatchify(
                tokens,
                self.x_embedder.patch_size,
                self.x_embedder.grid_size,
                self.out_channels,
            )  # (B, C, H, W)

        u = run_head(self.u_blocks, self.u_final_layer) if predict_u else None
        v = run_head(self.v_blocks, self.v_final_layer) if predict_v else None
        return u, v


# --------------------------------------------------------------------------------
# Transformer Blocks with RoPE Attention, SwiGLU, and Zero-init Residual Gates.
# --------------------------------------------------------------------------------
def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Rotate interleaved channel pairs of x by the position angles.

    x: (B, N, H, D)
    cos, sin: (N, D / 2)
    return: (B, N, H, D) in the dtype of x
    """
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)  # (B, N, H, D / 2) each
    cos, sin = cos[None, :, None], sin[None, :, None]  # [1, N, 1, D / 2] each
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with QK RMSNorm and RoPE; no biases."""
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, N, D)
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B, N, D)
        """
        B, N, D = x.shape
        q = self.q_norm(self.q_proj(x).view(B, N, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, N, self.num_heads, self.head_dim))
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, *rope) / math.sqrt(self.head_dim)
        k = apply_rotary_pos_emb(k, *rope)
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k)
        attn = attn.softmax(dim=-1, dtype=torch.float32)
        x = torch.einsum("bhqk,bkhd->bqhd", attn.to(v.dtype), v)
        return self.out_proj(x.reshape(B, N, D))


class SwiGLUMlp(nn.Module):
    """Swish-gated linear unit MLP; no biases."""
    def __init__(self, hidden_size: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, mlp_hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_size, mlp_hidden_dim, bias=False)
        self.w2 = nn.Linear(mlp_hidden_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class IMeanFlowDiTBlock(nn.Module):
    """
    Pre-norm transformer block without adaLN.

    - RMSNorm, RoPE attention, and SwiGLU MLP.
    - Each residual branch is gated by a zero-initialized per-channel scale, so the
      block starts as an identity mapping.
    """
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 8 / 3
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, N, D)
        rope: (cos, sin) of shape (N, head_dim / 2)
        return x: (B, N, D)
        """
        x = x + self.attn(self.norm1(x), rope) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer: RMSNorm, then a zero-initialized linear layer."""
    def __init__(
        self, hidden_size: int, patch_size: int | tuple[int, int], out_channels: int
    ) -> None:
        super().__init__()
        ph, pw = to_2tuple(patch_size)
        self.norm_final = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, ph * pw * out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        return x: (B, N, ph * pw * out_channels)
        """
        return self.linear(self.norm_final(x))


# --------------------------------------------------------------------------------
# IMeanFlowDiT Configs.
# --------------------------------------------------------------------------------
def _build_imeanflow_dit(cfg_model, **model_kwargs) -> IMeanFlowDiT:
    return IMeanFlowDiT(
        input_size=cfg_model.input_size,
        num_classes=cfg_model.num_classes,
        **model_kwargs
    )


def IMeanFlowDiT_XL_2(cfg_model, **kwargs) -> IMeanFlowDiT:
    return _build_imeanflow_dit(
        cfg_model, depth=48, hidden_size=1024, patch_size=2, num_heads=16, **kwargs
    )

def IMeanFlowDiT_L_2(cfg_model, **kwargs) -> IMeanFlowDiT:
    return _build_imeanflow_dit(
        cfg_model, depth=32, hidden_size=1024, patch_size=2, num_heads=16, **kwargs
    )

def IMeanFlowDiT_M_2(cfg_model, **kwargs) -> IMeanFlowDiT:
    return _build_imeanflow_dit(
        cfg_model, depth=24, hidden_size=768, patch_size=2, num_heads=12, **kwargs
    )

def IMeanFlowDiT_B_2(cfg_model, **kwargs) -> IMeanFlowDiT:
    return _build_imeanflow_dit(
        cfg_model, depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs
    )


IMeanFlowDiTModelNames = {
    "imeanflowdit_xl_2": IMeanFlowDiT_XL_2,
    "imeanflowdit_l_2":  IMeanFlowDiT_L_2,
    "imeanflowdit_m_2":  IMeanFlowDiT_M_2,
    "imeanflowdit_b_2":  IMeanFlowDiT_B_2,
}


# --------------------------------------------------------------------------------
# IMeanFlowDiT Model Class.
# --------------------------------------------------------------------------------
class IMeanFlowDiTModel(BaseModel):
    """
    FSDP-wrapped IMeanFlowDiT model.

    - Builds the configured IMeanFlowDiT variant.
    - Shards each IMeanFlowDiTBlock as an FSDP unit.
    - Inherits EMA and state-dict behavior from BaseModel.
    """

    def build_model(self) -> nn.Module:
        return IMeanFlowDiTModelNames[self.cfg_model.name](self.cfg_model)

    def wrap_modules(self) -> tuple:
        return (IMeanFlowDiTBlock,)
