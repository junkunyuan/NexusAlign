"""
Multi-reward: the sum of several component rewards, each divided by a fixed divisor.

References:
    - DiffusionART:
        - Official code: https://github.com/vanzll/DiffusionART
            - src/visual_rl/rewards/composite.py: CompositeImageReward, for the
              scores of all components
            - src/visual_rl/rewards/scalar.py: weighted_score, for the fixed
              normalization and the sum of the components
            - configs/examples/sd3_multireward.yaml: the divisor of each component
"""

import numpy as np
import torch

from nexus_align.registry import registry


def combine_group_zscores(
    prompts: list[str], scores: dict, weights: dict, eps: float = 1.0e-4
) -> np.ndarray:
    """
    Combine the components of a multi-reward into one training reward.

    Each component gets its group z-scores, with the eps after the population std.
    The training reward is the mean of these z-scores with the given weights. It is
    already normalized in each group, so an algorithm takes it as the advantages.
    """
    prompts = np.array(prompts)
    total = np.zeros(len(prompts))
    for key, weight in weights.items():
        rewards = np.asarray(scores[key], dtype=np.float64)
        for prompt in np.unique(prompts):
            group = rewards[prompts == prompt]
            zscores = (group - group.mean()) / (group.std() + eps)
            total[prompts == prompt] += weight * zscores
    return total / sum(weights.values())


class MultiReward:
    """
    Score each component reward, and divide each score by its divisor.

    - The headline score is the sum of the calibrated scores, as in DiffusionART.
    - weights holds the training weight of each component. An algorithm that supports
      a multi-reward trains on the weighted mean of the group z-scores of the
      calibrated scores, and not on the headline score.
    - metrics lists the components of configs/reward/multi_reward.yaml. It also holds
      the calibrated scores, so that the trainer gathers them.
    """

    metrics = (
        "multi_reward", "accuracy", "strict_accuracy", "geneval", "pickscore",
        "clipscore", "hpsv2", "geneval_calibrated", "pickscore_calibrated",
        "clipscore_calibrated", "hpsv2_calibrated",
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
            calibrated = torch.as_tensor(scores[reward.name], dtype=torch.float64)
            calibrated = calibrated / self.divisors[key]
            results.update(scores)
            results[f"{key}_calibrated"] = calibrated.tolist()
            total = total + calibrated
        results[self.name] = total.tolist()
        return results
