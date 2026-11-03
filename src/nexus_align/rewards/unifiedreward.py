"""
UnifiedReward reward: a LLaVA-OneVision judge scores an image from 1 to 5.

UnifiedReward runs in an sglang server. See scripts/tools/reward_server.sh.

References:
    - UnifiedReward:
        - Paper: Unified Reward Model for Multimodal Understanding and Generation
          (arXiv 2025)
        - Weights: https://huggingface.co/CodeGoat24/UnifiedReward-7b-v1.5
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
          (flow_grpo/rewards.py, unifiedreward_score_sglang)
"""

import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from nexus_align.rewards.remote import RemoteReward, to_uint8

# The exact question of Flow-GRPO.
QUESTION = (
    "<image>\nYou are given a text caption and a generated image based on that "
    "caption. Your task is to evaluate this image based on two key criteria:\n1. "
    "Alignment with the Caption: Assess how well this image aligns with the provided "
    "caption. Consider the accuracy of depicted objects, their relationships, and "
    "attributes as described in the caption.\n2. Overall Image Quality: Examine the "
    "visual quality of this image, including clarity, detail preservation, color "
    "accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a "
    "score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText "
    "Caption: [{prompt}]"
)
SCORE_RE = re.compile(r"Final Score:\s*([1-5](?:\.\d+)?)")


class UnifiedReward(RemoteReward):
    """
    Client of the sglang server. Each image of a batch is a concurrent chat request.

    - unifiedreward: the score from 1 to 5, divided by 5 for training, as Flow-GRPO
      does. If the answer has no score, the value is 0.
    """

    metrics = ("unifiedreward",)  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        # The session retries dropped connections, for example a pooled connection that
        # the server closed because it was idle. The judge decodes greedily, so a retry
        # gives the same score.
        super().__init__(cfg_reward)
        self.normalize = cfg_reward.normalize  # off for evaluation
        self.api_key = cfg_reward.api_key
        self.model = cfg_reward.model

    @staticmethod
    def encode(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image;base64,{encoded}"

    def score(self, image: Image.Image, prompt: str) -> float:
        """Ask the judge about one image. Temperature 0 decodes greedily."""
        response = self.session.post(
            f"{self.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.encode(image)}},
                        {"type": "text", "text": QUESTION.format(prompt=prompt)},
                    ],
                }],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        match = SCORE_RE.search(response.json()["choices"][0]["message"]["content"])
        scale = 5 if self.normalize else 1
        return float(match.group(1)) / scale if match else 0.0

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = [Image.fromarray(i).resize((512, 512)) for i in to_uint8(images)]
        with ThreadPoolExecutor(len(images)) as pool:
            scores = list(pool.map(self.score, images, prompts))
        return {"unifiedreward": scores}
