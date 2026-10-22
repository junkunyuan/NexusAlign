"""MLP components: feed-forward networks."""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Two-layer MLP with tanh-approximated GELU, adapted from timm."""
    def __init__(self, hidden_size: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
