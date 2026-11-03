"""
Embedding components: patch and timestep embeddings.

References:
    - DiT:
        - Paper: Scalable Diffusion Models with Transformers (ICCV 2023)
        - Official code: https://github.com/facebookresearch/DiT
            - models.py: TimestepEmbedder
    - timm:
        - Code: https://github.com/huggingface/pytorch-image-models
            - timm/layers/patch_embed.py: PatchEmbed
"""

import math

import torch
import torch.nn as nn


def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


class PatchEmbed(nn.Module):
    """
    Image to patch embedding, adapted from timm.

    - Splits the input into non-overlapping patches with a strided convolution.
    - Flattens the patches into a token sequence.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        H, W = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return x: (B, N, D)
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    """
    Scalar timestep embedder.

    - Builds sinusoidal frequency embeddings.
    - Projects them to the model hidden size with an MLP.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        t: (B,) a batch of timesteps, which can be fractional
        dim: the even dimension D of the output

        return: a (B, D) tensor of positional embeddings.

        Reference: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) a batch of timesteps
        t_emb: (B, D) a batch of timestep embeddings
        """
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size)
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
