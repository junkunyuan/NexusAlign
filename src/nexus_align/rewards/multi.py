"""Multi reward: the sum of several rewards, for multi-reward training.

References:
    - DiffusionNFT:
        - Official code: https://github.com/NVlabs/DiffusionNFT
          (flow_grpo/rewards.py, multi_score)
"""

import numpy as np

from nexus_align.registry import registry


class MultiReward:
    """
    Sum of the scores of the selected rewards under the key of this reward's name.

    - Every reward weighs 1.0, as DiffusionNFT; each one's score is the key of its name,
      e.g. PickScore divided by 26.
    - select(names) restricts the sum to some rewards, e.g. those of one training stage;
      the others are not computed.
    """

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.rewards = {
            name: registry.get("reward", cfg.name)(cfg_reward=cfg)
            for name, cfg in cfg_reward.rewards.items()
        }
        self.select(list(self.rewards))

    def select(self, names: list[str]) -> None:
        """Sum only the named rewards from now on."""
        unknown = [name for name in names if name not in self.rewards]
        if unknown:
            raise ValueError(f"❌ Rewards {unknown} are not in {list(self.rewards)}")
        self.selected = list(names)
        self.metrics = (self.name,) + tuple(
            metric for name in self.selected for metric in self.rewards[name].metrics
        )  # headline metrics to report

    def __call__(self, images, prompts, metadatas, only_strict=True) -> dict[str, list]:
        results, total = {}, np.zeros(len(prompts))
        for name in self.selected:
            scores = self.rewards[name](images, prompts, metadatas, only_strict)
            results.update(scores)
            total += np.asarray(scores[name], dtype=np.float64)
        results[self.name] = total.tolist()
        return results
