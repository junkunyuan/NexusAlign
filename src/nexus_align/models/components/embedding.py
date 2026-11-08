"""
Embedding components: patch, timestep, and 2D rotary position embeddings.

References:
    - DiT:
        - Paper: Scalable Diffusion Models with Transformers (ICCV 2023)
        - Official code: https://github.com/facebookresearch/DiT
            - models.py: TimestepEmbedder
    - timm:
        - Code: https://github.com/huggingface/pytorch-image-models
            - timm/layers/patch_embed.py: PatchEmbed
    - JiT:
        - Paper: Back to Basics: Let Denoising Generative Models Denoise (arXiv 2025)
        - Official code: https://github.com/LTH14/JiT
            - model_jit.py: BottleneckPatchEmbed
    - EVA-02:
        - Paper: EVA-02: A Visual Representation for Neon Genesis (arXiv 2023)
        - Official code: https://github.com/baaivision/EVA
            - EVA-02/asuka/rope.py: VisionRotaryEmbeddingFast
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


class BottleneckPatchEmbed(nn.Module):
    """
    Image to patch embedding through a low-dimensional bottleneck, from JiT.

    - A strided convolution maps each patch to bottleneck_dim channels.
    - A 1x1 convolution maps them to the hidden size.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        bottleneck_dim: int,
        hidden_size: int,
        proj1_bias: bool = True,
    ) -> None:
        super().__init__()
        H, W = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj1 = nn.Conv2d(
            in_channels, bottleneck_dim,
            kernel_size=self.patch_size, stride=self.patch_size, bias=proj1_bias,
        )
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return x: (B, N, D)
        """
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


def get_2d_rope_angles(head_dim: int, grid_size: tuple[int, int]) -> torch.Tensor:
    """
    Return the 2D RoPE angles of a patch grid, as EVA-02 makes them.

    The first half of the channel pairs rotates with the row of the patch, and the
    second half with its column.

    return: (gh * gw, head_dim / 2) angles of the patches in row-major order
    """
    half_dim = head_dim // 2
    freqs = torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim
    freqs = 1.0 / (10000 ** freqs)
    gh, gw = grid_size
    rows = torch.outer(torch.arange(gh, dtype=torch.float32), freqs)  # [gh, D / 4]
    cols = torch.outer(torch.arange(gw, dtype=torch.float32), freqs)  # [gw, D / 4]
    return torch.cat([
        rows[:, None].expand(gh, gw, -1), cols[None].expand(gh, gw, -1)
    ], dim=-1).reshape(gh * gw, half_dim)


class TimestepEmbedder(nn.Module):
    """
    Scalar timestep embedder.

    - Builds sinusoidal frequency embeddings.
    - Projects them to the model hidden size with an MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def positional_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        t: (B,) a batch of timesteps, which can be fractional
        dim: the even dimension D of the output
        max_period: the longest period of the frequencies

        return: a (B, D) tensor of positional embeddings.

        Reference: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
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
        t_freq = self.positional_embedding(
            t, dim=self.frequency_embedding_size, max_period=self.max_period
        )
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
