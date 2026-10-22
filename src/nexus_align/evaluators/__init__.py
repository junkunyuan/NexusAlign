"""Evaluator registry: register built-in evaluators by name on import."""

from nexus_align.registry import registry
from nexus_align.evaluators.fid_is_evaluator import FIDISEvaluator
from nexus_align.evaluators.reward_evaluator import RewardEvaluator

registry.register("evaluator", "in1k_50k_256_fid_is", FIDISEvaluator)
registry.register("evaluator", "reward", RewardEvaluator)
