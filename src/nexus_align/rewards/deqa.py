"""
DeQA reward: DeQA-Score image quality, a no-reference score of the image alone.

DeQA-Score runs in the DeQA server. See nexus_align/rewards/servers/deqa_server.py.

References:
    - DeQA-Score:
        - Paper: Teaching Large Language Models to Regress Accurate Image Quality Scores
          using Score Distribution (CVPR 2025)
        - Official code: https://github.com/zhiyuanyou/DeQA-Score
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/rewards.py, deqa_score_remote)
"""

import torch

from nexus_align.rewards.remote import RemoteReward, encode_image, to_uint8


class DeQA(RemoteReward):
    """
    Client of the DeQA server. It sends the images as JPEG bytes, as Flow-GRPO does.

    - deqa: the score from 1 to 5, divided by 5 for training, as Flow-GRPO does.
    """

    metrics = ("deqa",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        super().__init__(cfg_reward)
        self.normalize = cfg_reward.normalize  # off for evaluation

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = [encode_image(image, "JPEG") for image in to_uint8(images)]
        response = self.post({"images": images})
        scale = 5 if self.normalize else 1
        return {"deqa": [score / scale for score in response["scores"]]}
