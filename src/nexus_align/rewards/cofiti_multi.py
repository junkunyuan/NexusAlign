# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Adapted from the local finite-difference-flow-optimization/fdfo/{config,rewards,
# train_loop}.py. Licensed under the NVIDIA Source Code License v1 (Non-Commercial).
# See licenses/FDFO-LICENSE.txt for the full license.
"""The source multi_reward_geneval_clip30 preset and component-wise advantages."""

import copy
import math

import torch

from nexus_align.registry import registry


class CoFiTiMultiReward:
    """Report the fixed four-score sum; train on weighted component z-scores."""

    components = ("pickscore", "hpsv2", "clipscore", "geneval")
    scales = (1.0 / 26.0, 1.0, 1.0 / 30.0, 1.0)
    metrics = (
        "multi_reward_geneval_clip30", "pickscore", "hpsv2", "clipscore",
        "geneval", "accuracy", "strict_accuracy",
    )

    def __init__(self, cfg_reward):
        unknown = set(cfg_reward) - {
            "name", "weights", "std_epsilon", "clip_max", "rewards", "normalize",
        }
        if unknown:
            raise ValueError(f"Unsupported clip30 preset fields: {sorted(unknown)}")
        self.name = cfg_reward.name
        self.cfg_reward = cfg_reward
        if set(cfg_reward.weights) != set(self.components):
            raise ValueError("The clip30 preset requires four component weights")
        self.weights = {key: float(cfg_reward.weights[key]) for key in self.components}
        if not all(math.isfinite(w) for w in self.weights.values()):
            raise ValueError("Multi-reward weights must be finite")
        if not any(self.weights.values()):
            raise ValueError("At least one multi-reward weight must be nonzero")
        self.std_epsilon = float(cfg_reward.std_epsilon)
        self.clip_max = float(cfg_reward.clip_max)
        if not all(
            math.isfinite(v) and v > 0 for v in (self.std_epsilon, self.clip_max)
        ):
            raise ValueError("Reward epsilon and clip must be finite and positive")
        if set(cfg_reward.rewards) != set(self.components):
            raise ValueError("The preset requires PickScore, HPSv2, CLIP and GenEval")
        if cfg_reward.rewards.pickscore.normalize:
            raise ValueError("The clip30 preset calibrates raw PickScore exactly once")
        if cfg_reward.rewards.clipscore.score_divisor != 1:
            raise ValueError("The clip30 preset requires raw CLIP logits")
        self.rewards = {
            key: registry.get("reward", cfg_reward.rewards[key].name)(
                cfg_reward=cfg_reward.rewards[key]
            ) for key in self.components
        }
        self.metric_sentinels = self.rewards["geneval"].metric_sentinels

    @staticmethod
    def checked_scores(values, count, device):
        values = torch.as_tensor(values, dtype=torch.float32, device=device)
        if values.shape != (count,) or not bool(torch.isfinite(values).all()):
            raise ValueError("Reward components must be finite vectors matching images")
        return values

    @torch.no_grad()
    def __call__(self, images, prompts, metadatas, only_strict=True):
        count = len(prompts)
        if len(images) != count or len(metadatas) != count:
            raise ValueError("Multi-reward images, prompts and metadata must match")
        results = {}
        total = torch.zeros(count, dtype=torch.float32, device=images.device)
        weighted_total = torch.zeros_like(total)
        for key, scale in zip(self.components, self.scales):
            scores = self.rewards[key](
                images, prompts, copy.deepcopy(metadatas), only_strict=only_strict,
            )
            values = self.checked_scores(scores[key], count, images.device)
            calibrated = values * scale
            self.checked_scores(calibrated, count, images.device)
            results.update(scores)
            results[key + "_calibrated"] = calibrated.tolist()
            # Source reporting is unweighted, even for a zero training weight.
            total = total + calibrated
            weighted_total = weighted_total + self.weights[key] * calibrated
        self.checked_scores(total, count, images.device)
        self.checked_scores(weighted_total, count, images.device)
        results[self.name] = total.tolist()
        return results

    def training_rewards(self, scores, group_size, device):
        """Normalize each calibrated component, then sum without a final z-score."""
        count = len(scores[self.name])
        if group_size < 2 or count == 0 or count % group_size:
            raise ValueError("Multi-reward normalization requires complete groups")
        total = torch.zeros(count, dtype=torch.float32, device=device)
        for key in self.components:
            if self.weights[key] == 0:
                continue
            values = self.checked_scores(scores[key + "_calibrated"], count, device)
            grouped = values.reshape(-1, group_size)
            advantage = (
                (grouped - grouped.mean(1, keepdim=True))
                / (grouped.std(1, unbiased=False, keepdim=True) + self.std_epsilon)
            ).clamp(-self.clip_max, self.clip_max).flatten()
            total = total + self.weights[key] * advantage
            scores[key + "_group_zscore"] = advantage.tolist()
        self.checked_scores(total, count, device)
        scores["training_reward"] = total.tolist()
        return total
