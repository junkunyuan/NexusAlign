"""ImageReward reward: BLIP-based human preference score of an image for its prompt.

ImageReward runs in its server; see nexus_align/rewards/servers/imagereward_server.py.

References:
    - ImageReward:
        - Paper: ImageReward: Learning and Evaluating Human Preferences for
          Text-to-Image Generation
        - Official code: https://github.com/zai-org/ImageReward
"""

import torch

from nexus_align.rewards.remote import RemoteReward


class ImageReward(RemoteReward):
    """Client of the ImageReward server; images are sent as raw uint8 arrays."""

    metrics = ("imagereward",)  # headline metrics to report

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        response = self.post({"images": images, "prompts": list(prompts)})
        return {"imagereward": response["scores"]}
