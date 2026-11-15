"""
TwinTilt PickScore reward: PickScore of the float pixels, without uint8 rounding.

References:
    - FDFO:
        - Official code: https://github.com/NVlabs/finite-difference-flow-optimization
          (fdfo/rewards.py: RewardEngine._compute_pickscore)
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private, twintilt-twintilt.zip)
          (src/nexus_align/rewards/pickscore.py: the image_input "tensor" path)
"""

import torch

from nexus_align.rewards.pickscore import PickScore


class TwinTiltPickScore(PickScore):
    """PickScore that resizes the float pixels with antialiased bicubic, as FDFO."""

    metrics = ("twintilt_pickscore",)  # headline metrics to report

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = images.to(self.device, torch.float32)
        scores = self.score_grad(images, self.embed_prompts(prompts))
        return {"twintilt_pickscore": scores.tolist()}
