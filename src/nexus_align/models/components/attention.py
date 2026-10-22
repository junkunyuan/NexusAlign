"""Attention components: multi-head self-attention."""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention with QKV biases, adapted from timm."""
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        return x: (B, N, D)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, N, head_dim) each
        attn = (q * self.head_dim ** -0.5) @ k.transpose(-2, -1)
        x = attn.softmax(dim=-1) @ v
        return self.proj(x.transpose(1, 2).reshape(B, N, D))
