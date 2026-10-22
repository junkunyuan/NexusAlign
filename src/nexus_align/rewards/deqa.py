"""DeQA reward: DeQA-Score image quality, a no-reference score of the image alone.

DeQA-Score runs in the DeQA server; see nexus_align/rewards/servers/deqa_server.py.

References:
    - DeQA-Score:
        - Paper: Teaching Large Language Models to Regress Accurate Image Quality Scores
          using Score Distribution
        - Official code: https://github.com/zhiyuanyou/DeQA-Score
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/rewards.py, deqa_score_remote)
"""

from io import BytesIO

import torch
from PIL import Image

from nexus_align.rewards.remote import RemoteReward


class DeQA(RemoteReward):
    """
    Client of the DeQA server; images are sent as JPEG bytes, as Flow-GRPO does.

    - deqa: the score from 1 to 5, divided by 5 for training as Flow-GRPO.
    """

    metrics = ("deqa",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        super().__init__(cfg_reward)
        self.normalize = cfg_reward.get("normalize", True)  # off for evaluation

    @staticmethod
    def encode(image) -> bytes:
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        return buffer.getvalue()

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        response = self.post({"images": [self.encode(image) for image in images]})
        scale = 5 if self.normalize else 1
        return {"deqa": [score / scale for score in response["scores"]]}
