"""Reward registry: register built-in rewards by name on import."""

from nexus_align.registry import registry
from nexus_align.rewards.ocr import OCR
from nexus_align.rewards.deqa import DeQA
from nexus_align.rewards.hpsv2 import HPSv2
from nexus_align.rewards.hpsv3 import HPSv3
from nexus_align.rewards.geneval import GenEval
from nexus_align.rewards.multi import MultiReward
from nexus_align.rewards.aesthetic import Aesthetic
from nexus_align.rewards.clipscore import ClipScore
from nexus_align.rewards.pickscore import PickScore
from nexus_align.rewards.imagereward import ImageReward
from nexus_align.rewards.unifiedreward import UnifiedReward

registry.register("reward", "pickscore", PickScore)
registry.register("reward", "ocr", OCR)
registry.register("reward", "geneval", GenEval)
registry.register("reward", "aesthetic", Aesthetic)
registry.register("reward", "clipscore", ClipScore)
registry.register("reward", "hpsv2", HPSv2)
registry.register("reward", "imagereward", ImageReward)
registry.register("reward", "deqa", DeQA)
registry.register("reward", "unifiedreward", UnifiedReward)
registry.register("reward", "hpsv3", HPSv3)
registry.register("reward", "multi", MultiReward)
