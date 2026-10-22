"""HPSv3 reward: Qwen2-VL-7B human preference score of an image for its prompt.

HPSv3 runs in the HPSv3 server; see nexus_align/rewards/servers/hpsv3_server.py.

References:
    - HPSv3:
        - Paper: HPSv3: Towards Wide-Spectrum Human Preference Score
        - Official code: https://github.com/MizzenAI/HPSv3
"""

import torch

from nexus_align.rewards.remote import RemoteReward


class HPSv3(RemoteReward):
    """Client of the HPSv3 server; images are sent as raw uint8 arrays."""

    metrics = ("hpsv3",)  # headline metrics to report

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        response = self.post({"images": images, "prompts": list(prompts)})
        return {"hpsv3": response["scores"]}
