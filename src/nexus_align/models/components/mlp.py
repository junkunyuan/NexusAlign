"""
MLP components: feed-forward networks.

References:
    - timm:
        - Code: https://github.com/huggingface/pytorch-image-models
            - timm/layers/mlp.py: Mlp
"""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Two-layer MLP with GELU, adapted from timm. By default, GELU uses tanh."""
    def __init__(
        self, hidden_size: int, mlp_hidden_dim: int, approximate: str = "tanh"
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim)
        self.act = nn.GELU(approximate=approximate)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
