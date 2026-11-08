"""
PixelDiT2 model: a DiT on pixel patches that predicts the clean image, with the
grounding of a frozen DINOv3 encoder.

References:
    - PixelDiT2:
        - Paper: PixelDiT2: Representation-Grounded Pixel Diffusion Transformers
          (NeurIPS 2026)
        - Official code: https://github.com/NVlabs/PixelDiT
            - pixdit_core/pixeldit2_c2i.py: PixelDiT2, and the PixelDiT2_{B,L,H}_16
              configs
            - pixdit_core/grounding.py: GroundingConditioner, TimestepProjection
            - pixdit_core/modules.py: ModulatedDiTBlock, PatchFinalLayer
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_align.models.base_model import BaseModel
from nexus_align.models.components.dit import unpatchify
from nexus_align.models.components.embedding import TimestepEmbedder, to_2tuple
from nexus_align.models.pixeldit import (
    PixelDiTBlock, RMSNorm, apply_adaln, get_rope_angles, repa_projector
)


# --------------------------------------------------------------------------------
# Core PixelDiT2 Model.
# --------------------------------------------------------------------------------
class PixelDiT2(nn.Module):
    """
    Core architecture for PixelDiT2.

    - Embeds the pixel patches with a linear layer. A 2D RoPE gives the positions.
    - Grounding: a frozen DINOv3 encodes the noised image, and P_g maps its patch
      features to one grounding token per patch; see paper Eq. 4 and 5.
    - The condition of a patch is its grounding token plus the embeddings of t and
      the class label. The adaLN of each block uses it per patch (spatial adaLN).
    - Puts in_context_len class tokens before the patch tokens at block
      in_context_start, as JiT does. Their condition is the mean over the patches.
    - Predicts the clean image in the shape of the input.
    - A REPA projector maps the patch tokens of block repa_depth to repa_dim features.
    """
    def __init__(
        self,
        input_size: int | tuple[int, int] = 256,
        num_classes: int = 1000,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        in_context_len: int = 32,
        in_context_start: int = 8,
        proj_drop: float = 0.0,
        mlp_multiple: int = 0,
        single_silu_cond: bool = False,
        grounding_encoder: str = "",
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
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.single_silu_cond = single_silu_cond
        self.repa_depth = repa_depth

        # Build the modules in the official order, so that the random initialization
        # of the same seed matches.
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=10)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 null class
        self.s_embedder = nn.Linear(in_channels * patch_size ** 2, hidden_size)
        self.in_context_posemb = nn.Parameter(
            torch.zeros(1, in_context_len, hidden_size)
        )
        # Use dropout only in the middle half of the blocks.
        self.patch_blocks = nn.ModuleList([
            PixelDiTBlock(
                hidden_size, num_heads,
                proj_drop=proj_drop if depth // 4 <= i < depth // 4 * 3 else 0.0,
                mlp_multiple=mlp_multiple,
            )
            for i in range(depth)
        ])
        self.final_layer = PatchFinalLayer(hidden_size, patch_size, self.out_channels)

        # RoPE angles of the patches. The class tokens have zero angles, so RoPE does
        # not change them.
        angles = get_rope_angles(hidden_size // num_heads, self.grid_size)
        angles = torch.cat([torch.zeros(in_context_len, angles.shape[1]), angles])
        self.register_buffer("rope_cos", angles.cos(), persistent=False)  # [P + N, D/2]
        self.register_buffer("rope_sin", angles.sin(), persistent=False)  # [P + N, D/2]
        self.initialize_weights()

        self.grounding = GroundingEncoder(grounding_encoder, hidden_size, num_heads)
        if self.grounding.patch_size != patch_size:
            raise ValueError(
                f"❌ The grounding encoder has patch size {self.grounding.patch_size}, "
                f"but the model has {patch_size}"
            )
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

        # Initialize the positions of the class tokens:
        nn.init.normal_(self.in_context_posemb, std=0.02)

        # Set the adaLN modulation layers and the output layers to zero:
        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        drop_grounding: torch.Tensor | None = None,
        return_repa: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W) tensor of noised images
        t: (B,) tensor of timesteps; t = 1 is the clean image
        y: (B,) tensor of class labels
        drop_grounding: (B,) bool tensor of the samples without grounding tokens. If
            None, drop them for the null class, as the CFG unconditional branch does.
        return_repa: if true, also return the projected REPA tokens
        return: (B, C, H, W) tensor of predicted clean images, and with return_repa
            the (B, N, repa_dim) tensor of REPA tokens
        """
        P = self.in_context_len
        N = self.grid_size[0] * self.grid_size[1]
        rope_all = (self.rope_cos, self.rope_sin)
        rope = (self.rope_cos[P:], self.rope_sin[P:])
        t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        y_emb = self.y_embedder(y).unsqueeze(1)  # (B, 1, D)

        # Condition each patch on its grounding token, t, and the class label. The
        # samples without grounding use t and the class label only.
        c = self.grounding(x, t, rope) + t_emb + y_emb  # (B, N, D)
        c_null = t_emb + y_emb  # (B, 1, D)
        if not self.single_silu_cond:
            c, c_null = F.silu(c), F.silu(c_null)
        if drop_grounding is None:
            drop_grounding = y == self.num_classes
        c = torch.where(drop_grounding.view(-1, 1, 1), c_null.expand(-1, N, -1), c)
        c_all = torch.cat([c.mean(dim=1, keepdim=True).expand(-1, P, -1), c], dim=1)

        patches = F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)
        s = self.s_embedder(patches)  # (B, N, D)
        for i, block in enumerate(self.patch_blocks):
            if i == self.in_context_start:
                tokens = y_emb.expand(-1, P, -1) + self.in_context_posemb.to(s.dtype)
                s = torch.cat([tokens, s], dim=1)  # (B, P + N, D)
            has_tokens = i >= self.in_context_start
            s = block(s, c_all if has_tokens else c, rope_all if has_tokens else rope)
            if i + 1 == self.repa_depth:
                z = s[:, P:] if has_tokens else s

        x = self.final_layer(s[:, P:], c)  # (B, N, p * p * C)
        x = unpatchify(x, self.patch_size, self.grid_size, self.out_channels)
        if return_repa:
            return x, self.repa_proj(z)
        return x


# --------------------------------------------------------------------------------
# Grounding Encoder and Final Layer.
# --------------------------------------------------------------------------------
class GroundingEncoder(nn.Module):
    """
    Frozen DINOv3 encoder and the timestep-conditioned projection P_g.

    - DINOv3 encodes the noised image in [-1, 1] with the ImageNet normalization,
      and gives one feature per patch. It stays in eval mode and gets no gradients.
    - P_g lifts the features to the hidden size, runs one DiT block with the adaLN
      of t, and projects the output to the grounding tokens.
    - encoder_path is a directory of timm weights. Its name is the timm model name.
    """
    def __init__(self, encoder_path: str, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        import timm  # import it here, so that the other models do not need timm

        self.encoder = timm.create_model(
            os.path.basename(os.path.normpath(encoder_path)),
            pretrained=True,
            pretrained_cfg_overlay=dict(
                file=os.path.join(encoder_path, "model.safetensors")
            ),
            dynamic_img_size=True,
        )
        self.encoder.requires_grad_(False)
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.input_proj = nn.Linear(self.encoder.num_features, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=10)
        self.block = PixelDiTBlock(hidden_size, num_heads)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.block.adaLN_modulation[-1].bias, 0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def train(self, mode: bool = True) -> "GroundingEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, t: torch.Tensor, rope: tuple) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor of noised images in [-1, 1]
        t: (B,) tensor of timesteps
        rope: (cos, sin) of shape (N, head_dim / 2)
        return: (B, N, D) tensor of grounding tokens
        """
        x = (x + 1.0) * 0.5
        x = (x - self.image_mean.to(x.dtype)) / self.image_std.to(x.dtype)
        with torch.no_grad():
            features = self.encoder.forward_features(x)
        features = features[:, self.encoder.num_prefix_tokens:]  # (B, N, d_E)
        g = self.input_proj(features.to(self.input_proj.weight.dtype))
        c = self.t_embedder(t).unsqueeze(1).to(g.dtype)  # (B, 1, D)
        return self.output_proj(self.block(g, c, rope))


class PatchFinalLayer(nn.Module):
    """Final projection layer: RMSNorm with adaLN per token, then a linear layer."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size ** 2 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        c: (B, N, D)
        return x: (B, N, p * p * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(apply_adaln(self.norm(x), shift, scale))


# --------------------------------------------------------------------------------
# PixelDiT2 Configs.
# --------------------------------------------------------------------------------
PixelDiT2ModelNames = {
    "pixeldit2_h_16": dict(
        depth=32, hidden_size=1280, patch_size=16, num_heads=16, in_context_start=8,
        proj_drop=0.2, repa_depth=8,
    ),
    "pixeldit2_l_16": dict(
        depth=24, hidden_size=1024, patch_size=16, num_heads=16, in_context_start=8,
        repa_depth=8,
    ),
    "pixeldit2_b_16": dict(
        depth=12, hidden_size=768, patch_size=16, num_heads=12, in_context_start=4,
        repa_depth=4,
    ),
}


# --------------------------------------------------------------------------------
# PixelDiT2 Model Class.
# --------------------------------------------------------------------------------
class PixelDiT2Model(BaseModel):
    """
    FSDP-wrapped PixelDiT2 model.

    - Builds the configured PixelDiT2 variant with its frozen DINOv3 encoder.
    - Shards each PixelDiTBlock as an FSDP unit.
    - The EMA tracks only the trainable weights.
    """

    def build_model(self) -> nn.Module:
        return PixelDiT2(
            input_size=self.cfg_model.input_size,
            num_classes=self.cfg_model.num_classes,
            mlp_multiple=self.cfg_model.mlp_multiple,
            single_silu_cond=self.cfg_model.single_silu_cond,
            grounding_encoder=self.cfg_model.grounding_encoder,
            repa_dim=self.cfg_model.repa_dim,
            **PixelDiT2ModelNames[self.cfg_model.name],
        )

    def wrap_modules(self) -> tuple:
        return (PixelDiTBlock,)

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Update only the trainable EMA parameters, and not the frozen DINOv3."""
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            if p.requires_grad:
                ema_p.mul_(decay).add_(p.data, alpha=1 - decay)
