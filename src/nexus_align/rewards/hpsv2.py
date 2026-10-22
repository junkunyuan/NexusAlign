"""HPSv2.1 reward: HPS v2.1 CLIP-H human preference score of an image for its prompt.

HPSv2.1 runs in the HPSv2 server; see nexus_align/rewards/servers/hpsv2_server.py.

References:
    - HPSv2:
        - Paper: Human Preference Score v2: A Solid Benchmark for Evaluating Human
          Preferences of Text-to-Image Synthesis
        - Official code: https://github.com/tgxs002/HPSv2
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/hpsv2_scorer.py)
"""

import torch

from nexus_align.rewards.remote import RemoteReward


class HPSv2(RemoteReward):
    """Client of the HPSv2 server; images are sent as raw uint8 arrays."""

    metrics = ("hpsv2",)  # headline metrics to report

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        response = self.post({"images": images, "prompts": list(prompts)})
        return {"hpsv2": response["scores"]}
