"""
TwinTilt multi-reward: PickScore, HPSv2.1, ClipScore, and GenEval, each calibrated.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private, twintilt-twintilt.zip)
          (src/nexus_align/rewards/twintilt_multi.py: TwinTiltMultiReward.__call__; the
          multi_reward_geneval_clip30 preset)
"""

import torch

from nexus_align.registry import registry


class TwinTiltMultiReward:
    """
    Score each component reward, and divide each score by its divisor.

    - The headline score is the unweighted sum of the calibrated scores, as TwinTilt
      reports it.
    - weights holds the training weight of each component. The algorithm trains on the
      weighted mean of the group z-scores of the calibrated scores.
    """

    metrics = (
        "twintilt_multi", "twintilt_pickscore", "hpsv2", "clipscore", "accuracy",
        "strict_accuracy", "geneval",
    )  # headline metrics to report

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.divisors = dict(cfg_reward.divisors)
        self.weights = dict(cfg_reward.weights)
        self.rewards = {
            key: registry.get("reward", cfg.name)(cfg_reward=cfg)
            for key, cfg in cfg_reward.rewards.items()
        }

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        results, total = {}, 0
        for key, reward in self.rewards.items():
            scores = reward(images, prompts, metadatas, only_strict)
            calibrated = torch.as_tensor(scores[reward.name], dtype=torch.float32)
            calibrated = calibrated / self.divisors[key]
            results.update(scores)
            results[f"{key}_calibrated"] = calibrated.tolist()
            total = total + calibrated
        results[self.name] = total.tolist()
        return results
