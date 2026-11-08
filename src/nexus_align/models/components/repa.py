"""
REPA encoder: the frozen DINOv2 features that REPA aligns the model tokens with.

References:
    - REPA:
        - Paper: Representation Alignment for Generation: Training Diffusion
          Transformers Is Easier Than You Think (ICLR 2025)
        - Official code: https://github.com/sihyun-yu/REPA
    - PixelDiT2:
        - Official code: https://github.com/NVlabs/PixelDiT
            - c2i/src/grounded_diffusion.py: REPATeacher, _dinov2_pos_embed
    - DINOv2:
        - Official code: https://github.com/facebookresearch/dinov2
            - dinov2/models/vision_transformer.py: interpolate_pos_encoding
"""

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 of timm with bf16 weights, for images in [-1, 1].

    - Resizes the images, so that DINOv2 gives one token per model patch.
    - Resamples the position embedding as the official DINOv2 code does. timm
      resamples it in a different way.
    - encoder_path is a directory of timm weights. Its name is the timm model name.
    """
    def __init__(self, encoder_path: str) -> None:
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
        self.encoder.requires_grad_(False).eval()
        self.encoder.to(torch.bfloat16)
        self.encoder._pos_embed = types.MethodType(dinov2_pos_embed, self.encoder)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor of images in [-1, 1]
        patch_size: patch size of the model
        return: (B, N, C_E) bf16 tensor of patch features, where N = H W / patch_size^2
        """
        x = normalize((x + 1.0) / 2.0, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        H, W = x.shape[-2:]
        p = self.encoder.patch_embed.patch_size[0]
        x = F.interpolate(
            x, (int(p * H / patch_size), int(p * W / patch_size)), mode="bicubic"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.encoder.forward_features(x)
        return features[:, self.encoder.num_prefix_tokens:].to(torch.bfloat16)


def dinov2_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
    """
    Add the position embedding to the patches, and put the prefix tokens first.

    This replaces timm's _pos_embed. The bicubic resampling uses the scale
    (size + 0.1) / grid and no antialias, as DINOv2's interpolate_pos_encoding does.

    x: (B, H, W, C) tensor of patch embeddings
    return: (B, prefix + H W, C)
    """
    B, H, W, C = x.shape
    grid = self.patch_embed.grid_size[0]
    pos = self.pos_embed.float()
    prefix_pos = pos[:, :self.num_prefix_tokens]
    patch_pos = pos[:, self.num_prefix_tokens:].reshape(1, grid, grid, C)
    patch_pos = patch_pos.permute(0, 3, 1, 2)
    if (H, W) != (grid, grid):
        scale = ((W + 0.1) / grid, (H + 0.1) / grid)
        patch_pos = F.interpolate(
            patch_pos, scale_factor=scale, mode="bicubic", antialias=False
        )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, C)
    pos = torch.cat([prefix_pos, patch_pos], dim=1).to(x.dtype)
    prefix = [
        token.expand(B, -1, -1)
        for token in (self.cls_token, getattr(self, "reg_token", None))
        if token is not None
    ]
    x = torch.cat(prefix + [x.view(B, -1, C)], dim=1)
    return self.pos_drop(x + pos)
