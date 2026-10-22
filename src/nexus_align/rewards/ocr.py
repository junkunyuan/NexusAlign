"""OCR reward: 1 - normalized edit distance between the quoted text and the OCR result.

PaddleOCR runs in the OCR server; see nexus_align/rewards/servers/ocr_server.py.

References:
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
"""

import torch

from nexus_align.rewards.remote import RemoteReward


class OCR(RemoteReward):
    """Client of the OCR server; images are sent as raw uint8 arrays."""

    metrics = ("ocr",)  # headline metrics to report

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        response = self.post({"images": images, "prompts": list(prompts)})
        return {"ocr": response["scores"]}
