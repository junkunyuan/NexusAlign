"""
TwinTilt multi-reward: PickScore, HPSv2.1, ClipScore, and GenEval, each calibrated.

It reuses MultiReward, and reports the float-pixel PickScore of TwinTilt as its
PickScore component.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private, twintilt-twintilt.zip)
            - src/nexus_align/rewards/twintilt_multi.py: TwinTiltMultiReward.__call__;
              the multi_reward_geneval_clip30 preset
"""

from nexus_align.rewards.multi_reward import MultiReward


class TwinTiltMultiReward(MultiReward):
    """MultiReward whose PickScore component scores float pixels, as TwinTilt does."""

    metrics = (
        "twintilt_multi", "accuracy", "strict_accuracy", "geneval",
        "twintilt_pickscore", "clipscore", "hpsv2", "geneval_calibrated",
        "pickscore_calibrated", "clipscore_calibrated", "hpsv2_calibrated",
    )  # headline metrics to report
