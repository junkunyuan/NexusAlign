"""
HPSv3 reward: human preference score of an image for its prompt, with Qwen2-VL-7B.

HPSv3 runs in the HPSv3 server. See nexus_align/rewards/servers/hpsv3_server.py.

References:
    - HPSv3:
        - Paper: HPSv3: Towards Wide-Spectrum Human Preference Score (ICCV 2025)
        - Official code: https://github.com/MizzenAI/HPSv3
"""

from nexus_align.rewards.remote import RemoteReward


class HPSv3(RemoteReward):
    """Client of the HPSv3 server. It sends the images as raw uint8 arrays."""

    metrics = ("hpsv3",)  # headline metrics to report
