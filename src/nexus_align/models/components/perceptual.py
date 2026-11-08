"""
Perceptual losses: LPIPS with VGG16, and the distance of ConvNeXt-V2 features.

References:
    - pMeanFlow:
        - Official code: https://github.com/Lyy-iiis/pMF
            - utils/auxloss_util.py: paired_random_resized_crop, init_auxloss
            - models/convnext.py: ConvNextV2.forward_features
    - LPIPS:
        - Official code: https://github.com/richzhang/PerceptualSimilarity
            - lpips/lpips.py: LPIPS, ScalingLayer, normalize_tensor
            - lpips/pretrained_networks.py: vgg16
    - JAX:
        - Official code: https://github.com/jax-ml/jax
            - jax/_src/image/scale.py: compute_weight_mat, _fill_keys_cubic_kernel
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import ConvNextV2Model


class LPIPS(nn.Module):
    """
    LPIPS (v0.1) of VGG16 features, for images in [-1, 1].

    - Normalizes each VGG16 feature map over the channels at five ReLU layers.
    - Weights the squared differences per channel, and averages them over space.
    """
    def __init__(self, vgg_path: str, lpips_path: str) -> None:
        super().__init__()
        vgg = torchvision.models.vgg16()
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        bounds = [0, 4, 9, 16, 23, 30]  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slices = nn.ModuleList(
            vgg.features[start:end] for start, end in zip(bounds[:-1], bounds[1:])
        )
        lins = torch.load(lpips_path, map_location="cpu")
        for i in range(len(self.slices)):
            self.register_buffer(f"lin{i}", lins[f"lin{i}.model.1.weight"])
        shift = torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        scale = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)
        self.register_buffer("shift", shift)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, 3, H, W) images in [-1, 1]; only x gets gradients
        return: (B,) LPIPS distances
        """
        def normalize(feat):
            return feat / (feat.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)

        x = (x - self.shift) / self.scale
        with torch.no_grad():
            y = (y - self.shift) / self.scale
            feats_y = []
            for block in self.slices:
                y = block(y)
                feats_y.append(normalize(y))
        dist = 0
        for i, block in enumerate(self.slices):
            x = block(x)
            diff = (normalize(x) - feats_y[i]).square()
            dist = dist + F.conv2d(diff, getattr(self, f"lin{i}")).mean(dim=(1, 2, 3))
        return dist


class ConvNeXtDistance(nn.Module):
    """
    Squared distance of the pooled ConvNeXt-V2 features, before the classifier head.

    The official code gives the images in [-1, 1] to ConvNeXt-V2 without the ImageNet
    normalization, so this module does the same.
    """
    def __init__(self, convnext_path: str) -> None:
        super().__init__()
        self.net = ConvNextV2Model.from_pretrained(convnext_path)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, 3, H, W) images in [-1, 1]; only x gets gradients
        return: (B,) squared feature distances
        """
        with torch.no_grad():
            feats_y = self.net(y).pooler_output
        return (self.net(x).pooler_output - feats_y).square().sum(dim=1)


def resize_weights(
    in_size: int, out_size: int, scale: float, translation: float, device
) -> torch.Tensor:
    """
    Return the (in_size, out_size) weights of one axis of a cubic resize.

    This is compute_weight_mat of jax.image.scale_and_translate with the Keys cubic
    kernel and antialias. Output pixel o samples the input at (o + 0.5 - translation)
    / scale - 0.5, and the kernel can also read the pixels outside the crop.
    """
    inv_scale = 1.0 / scale
    sample = torch.arange(out_size, device=device) + 0.5 - translation
    sample = sample * inv_scale - 0.5
    dist = (sample[None] - torch.arange(in_size, device=device)[:, None]).abs()
    x = dist / max(inv_scale, 1.0)
    weights = torch.where(
        x >= 1, ((-0.5 * x + 2.5) * x - 4) * x + 2, ((1.5 * x - 2.5) * x) * x + 1
    )
    weights = torch.where(x >= 2, 0.0, weights)
    total = weights.sum(dim=0, keepdim=True)
    weights = torch.where(total.abs() > 1000 * torch.finfo(torch.float32).eps,
                          weights / torch.where(total != 0, total, 1), 0.0)
    inside = (sample >= -0.5) & (sample <= in_size - 0.5)
    return torch.where(inside[None], weights, 0.0)


def paired_random_resized_crop(
    x: torch.Tensor, y: torch.Tensor, size: int = 224
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop the same random box of each pair, and resize it to size x size.

    The box has a scale in [0.08, 1] and an aspect ratio in [3/4, 4/3], as in the
    official paired_random_resized_crop. The resize multiplies with the weights of
    resize_weights, so its backward pass is deterministic.
    """
    _, _, H, W = x.shape
    weights_h, weights_w = [], []
    for _ in range(x.shape[0]):
        area = H * W * random.uniform(0.08, 1.0)
        aspect = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        w = min(max(round(math.sqrt(area * aspect)), 1), W)
        h = min(max(round(math.sqrt(area / aspect)), 1), H)
        top, left = random.randint(0, H - h), random.randint(0, W - w)
        weights_h.append(resize_weights(H, size, size / h, -top * size / h, x.device))
        weights_w.append(resize_weights(W, size, size / w, -left * size / w, x.device))
    weights_h, weights_w = torch.stack(weights_h), torch.stack(weights_w)

    def resize(img):
        return torch.einsum("bhi,bchw,bwj->bcij", weights_h, img, weights_w)

    return resize(x), resize(y)
