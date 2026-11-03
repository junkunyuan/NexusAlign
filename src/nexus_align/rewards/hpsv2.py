"""
HPSv2.1 reward: human preference score of an image for its prompt, with CLIP-H.

HPSv2.1 runs in the HPSv2 server. See nexus_align/rewards/servers/hpsv2_server.py.

References:
    - HPSv2:
        - Paper: Human Preference Score v2: A Solid Benchmark for Evaluating Human
          Preferences of Text-to-Image Synthesis (arXiv 2023)
        - Official code: https://github.com/tgxs002/HPSv2
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/hpsv2_scorer.py)
"""

from nexus_align.rewards.remote import RemoteReward


class HPSv2(RemoteReward):
    """Client of the HPSv2 server. It sends the images as raw uint8 arrays."""

    metrics = ("hpsv2",)  # headline metrics to report
