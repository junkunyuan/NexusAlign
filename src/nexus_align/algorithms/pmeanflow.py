"""
pMeanFlow algorithm: the iMeanFlow v-loss on pixels, plus perceptual losses.

References:
    - pMeanFlow:
        - Paper: One-step Latent-free Image Generation with Pixel Mean Flows
          (arXiv 2026)
        - Official code: https://github.com/Lyy-iiis/pMF
            - pmf.py: pixelMeanFlow.forward, sample_tr, generate
            - utils/auxloss_util.py: init_auxloss
"""

import torch

from nexus_align.algorithms.imeanflow import IMeanFlow
from nexus_align.models.components.perceptual import (
    LPIPS, ConvNeXtDistance, paired_random_resized_crop
)


class PMeanFlow(IMeanFlow):
    """
    pMeanFlow training algorithm.

    The model predicts the denoised image x and returns it as a velocity (see
    PMeanFlowDiT), so the loss is the iMeanFlow loss. pMeanFlow also scales the noise,
    samples some (r, t) uniformly, and adds the perceptual losses of x.
    """

    def __init__(self, cfg_algorithm):
        super().__init__(cfg_algorithm)

        # Noise configs.
        self.noise_scale = cfg_algorithm.get("noise_scale", 1.0)  # noise std; image_size / 256 in paper Table 8
        self.uniform_time_prob = cfg_algorithm.get("uniform_time_prob", 0.0)  # probability to sample r and t uniformly

        # Perceptual loss configs.
        self.lpips_weight = cfg_algorithm.get("lpips_weight", 0.4)  # weight of the LPIPS loss; 0 disables it
        self.convnext_weight = cfg_algorithm.get("convnext_weight", 0.1)  # weight of the ConvNeXt-V2 loss; 0 disables it
        self.perceptual_max_t = cfg_algorithm.get("perceptual_max_t", 0.8)  # use the perceptual losses only for t < perceptual_max_t
        self.vgg_path = cfg_algorithm.get("vgg_path", None)  # VGG16 weights of LPIPS
        self.lpips_path = cfg_algorithm.get("lpips_path", None)  # LPIPS v0.1 linear weights of VGG16
        self.convnext_path = cfg_algorithm.get("convnext_path", None)  # ConvNeXt-V2 model path

        # Training builds the perceptual networks at the first forward pass.
        self.lpips = None
        self.convnext = None

    def sample_r_t(self, batch_size):
        """
        Sample the time steps (r, t), with r = t for the first samples.

        Unlike iMeanFlow, the official code sets r = t before it sorts r and t, so t of
        a sample with r = t has one logit-normal draw, not the max of two.
        """
        def logit_normal():
            noise = torch.randn(batch_size, device=self.device)
            return torch.sigmoid(noise * self.time_sigma + self.time_mu)

        t, r = logit_normal(), logit_normal()
        uniform = torch.rand(batch_size, device=self.device) < self.uniform_time_prob
        t = torch.where(uniform, torch.rand(batch_size, device=self.device), t)
        r = torch.where(uniform, torch.rand(batch_size, device=self.device), r)
        num_equal = int(batch_size * (1.0 - self.ratio_r_not_equal_t))
        r = torch.where(torch.arange(batch_size, device=self.device) < num_equal, t, r)
        return torch.minimum(r, t), torch.maximum(r, t)

    def add_noise(self, z_0, t):
        """
        Return the noised image z_t and the conditional velocity at t.

        The velocity is (z_t - z_0) / t, with t clipped to [0.05, 1] as in the
        official code, so it is smaller than noise - z_0 for t < 0.05.
        """
        t = t.view(-1, 1, 1, 1)
        noises = torch.randn_like(z_0) * self.noise_scale
        z_t = (1 - t) * z_0 + t * noises
        return z_t, (z_t - z_0) / t.clamp(0.05, 1.0)

    def aux_loss(self, z_0, z_t, t, u):
        """
        Return the adaptively weighted perceptual losses of the denoised images.

        The losses compare x = z_t - t * u with the clean image z_0, on the same random
        crop of each pair. As in the official code, they run on all samples and keep
        only the samples with t < perceptual_max_t. Fixed shapes also keep cuDNN from
        planning the convolutions again at every step.
        """
        if self.lpips is None and self.lpips_weight > 0:
            self.lpips = LPIPS(self.vgg_path, self.lpips_path).to(self.device)
            self.lpips.requires_grad_(False).eval()
        if self.convnext is None and self.convnext_weight > 0:
            self.convnext = ConvNeXtDistance(self.convnext_path).to(self.device)
            self.convnext.requires_grad_(False).eval()

        x = z_t - t.view(-1, 1, 1, 1) * u
        x, x_gt = paired_random_resized_crop(x, z_0)
        keep = t < self.perceptual_max_t
        loss = torch.zeros_like(t)
        metrics = {}
        for name, net, weight in (
            ("lpips", self.lpips, self.lpips_weight),
            ("convnext", self.convnext, self.convnext_weight),
        ):
            if weight > 0:
                dist = torch.where(keep, net(x, x_gt), 0.0)
                loss = loss + weight * self.adaptive_weight(dist)
                metrics[f"loss_{name}"] = dist.detach().mean()
        return loss, metrics

    @torch.no_grad()
    def sample(self, data):
        """Generate samples as iMeanFlow does, from noise with std noise_scale."""
        return super().sample({**data, "latents": data["latents"] * self.noise_scale})
