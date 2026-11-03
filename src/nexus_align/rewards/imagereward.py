"""
ImageReward reward: human preference score of an image for its prompt, with BLIP.

ImageReward runs in its server. See nexus_align/rewards/servers/imagereward_server.py.

References:
    - ImageReward:
        - Paper: ImageReward: Learning and Evaluating Human Preferences for
          Text-to-Image Generation (NeurIPS 2023)
        - Official code: https://github.com/zai-org/ImageReward
"""

from nexus_align.rewards.remote import RemoteReward


class ImageReward(RemoteReward):
    """Client of the ImageReward server. It sends the images as raw uint8 arrays."""

    metrics = ("imagereward",)  # headline metrics to report
