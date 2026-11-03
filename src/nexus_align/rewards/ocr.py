"""
OCR reward: 1 - normalized edit distance between the quoted text and the OCR result.

PaddleOCR runs in the OCR server. See nexus_align/rewards/servers/ocr_server.py.

References:
    - Flow-GRPO:
        - Official code: https://github.com/yifan123/flow_grpo
"""

from nexus_align.rewards.remote import RemoteReward


class OCR(RemoteReward):
    """Client of the OCR server. It sends the images as raw uint8 arrays."""

    metrics = ("ocr",)  # headline metrics to report
