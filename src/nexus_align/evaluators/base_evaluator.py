"""Base evaluator: abstract interface and the unified checkpoint sweep."""

import os
import re
import glob
from abc import ABC, abstractmethod
from collections.abc import Mapping

import yaml
import torch.distributed as dist
from omegaconf import OmegaConf

from nexus_align.eval.summary import refresh_summary
from nexus_align.models.base_model import BaseModel
from nexus_align.engine.checkpoint import CheckpointManager
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


# Configs that select only what to sweep or keep, not how to calculate the metrics.
SWEEP_KEYS = ("ckpts", "dry_run", "new_run", "save_images", "num_workers", "compile")

CKPT_RE = re.compile(r"(?:total_step_)?(\d+)$")
CKPT_STATE_RE = re.compile(r"epoch_(\d+)-step_(\d+)-total_step_(\d+)$")


def parse_step(ckpt: str) -> int | None:
    """Return the total training step encoded in a checkpoint name."""
    match = CKPT_RE.search(os.path.basename(ckpt))
    return int(match.group(1)) if match else None


def parse_train_state(ckpt: str) -> dict[str, int | None]:
    """Return the epoch, step, and total step encoded in a checkpoint name."""
    match = CKPT_STATE_RE.search(os.path.basename(ckpt))
    if match is None:
        return {"epoch": None, "step": None, "total_step": parse_step(ckpt)}
    epoch, step, total_step = (int(g) for g in match.groups())
    return {"epoch": epoch, "step": step, "total_step": total_step}


class BaseEvaluator(ABC):
    """
    Base evaluator that does the checkpoint sweep and aggregates the metrics.

    - __init__ prepares the paths, the checkpoint manager, and the evaluated weights
      from the evaluator config.
    - run() evaluates every checkpoint without metrics yet, then refreshes the
      summary JSON file and the evaluation curves after each new result.
    - Subclasses provide the per-checkpoint evaluation.
    """

    def __init__(
        self,
        cfg_evaluator: Mapping | None = None,
        model: BaseModel = None,
        algorithm: BaseAlgorithm = None,
        log_dir: str = "logs/temp",
    ) -> None:
        # Get the model, the algorithm, and the evaluation config.
        self.model = model
        self.algorithm = algorithm
        for key, value in cfg_evaluator.items():
            setattr(self, key, value)

        # Get the checkpoint directory of the experiment. prepare_env sets eval_dir.
        self.ckpt_dir = os.path.join(log_dir, "checkpoints")
        if dist.get_rank() == 0:
            config_path = os.path.join(self.eval_dir, "config_eval.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = yaml.safe_load(f)["evaluator"]
            config = OmegaConf.to_container(cfg_evaluator, resolve=True)
            self.check_config(saved_config, config)
        dist.barrier()

        # Build the checkpoint manager, which loads only the evaluated weights.
        self.checkpoint_manager = CheckpointManager(save_ckpt_dir=self.ckpt_dir)
        self.load_ckpt_keys = [self.model_key, "algorithm"]

        # Sample with the configured weights. The algorithm runs the sampling loop.
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = model.device
        self.eval_model = getattr(model, self.model_key)
        assert self.eval_model is not None, (
            f"❌ Model has no <{self.model_key}> weights to evaluate"
        )
        self.eval_model.eval()

    def check_config(self, saved_config: dict, config: dict) -> None:
        """Warn if a reused evaluation directory used a different evaluator config."""
        changed = [k for k, v in config.items()
                   if k not in SWEEP_KEYS and saved_config.get(k) != v]
        if changed:
            print(f"⚠️ WARNING: reusing <{self.eval_dir}> built with different configs: "
                  f"{changed}; set evaluator.new_run=true to evaluate in a new dir")

    def find_ckpts(self) -> list[str]:
        """List the checkpoints to evaluate, sorted by training step."""
        ckpts = list(self.ckpts) if self.ckpts else glob.glob(os.path.join(self.ckpt_dir, "epoch_*"))
        ckpts = [c for c in ckpts if parse_step(c) is not None]
        return sorted(ckpts, key=parse_step)

    def refresh_summary(self, ckpts: list[str]) -> None:
        """Aggregate the metrics of all evaluated checkpoints."""
        if dist.get_rank() == 0:
            refresh_summary(
                ckpt_metrics=[(parse_train_state(c), c, self.metrics_path(c)) for c in ckpts],
                out_dir=self.eval_dir,
            )
        dist.barrier()

    def ckpt_eval_dir(self, ckpt: str) -> str:
        """Return the evaluation directory of one checkpoint."""
        return os.path.join(self.eval_dir, os.path.basename(ckpt))

    def metrics_path(self, ckpt: str) -> str:
        """Return the metrics file of one checkpoint."""
        return os.path.join(self.ckpt_eval_dir(ckpt), "metrics.json")

    def is_evaluated(self, ckpt: str) -> bool:
        """Return True if the checkpoint has a metrics file, so it needs no evaluation."""
        return os.path.exists(self.metrics_path(ckpt))

    @abstractmethod
    def evaluate(self, ckpt: str) -> None:
        """Evaluate the currently loaded checkpoint and save its metrics."""
        ...

    # ----------------------------------------
    # Evaluation loop.
    # ----------------------------------------
    def run(self) -> None:
        """Evaluate every checkpoint that has no metrics yet."""
        ckpts = self.find_ckpts()
        if not ckpts:
            print(f"⚠️ WARNING: no checkpoint found in <{self.ckpt_dir}>")
            return

        pending = [c for c in ckpts if not self.is_evaluated(c)]
        info = [f"📌 Found {len(ckpts)} checkpoints, {len(pending)} of them to evaluate"]
        for i, ckpt in enumerate(pending):
            info += [f"{i}. <{ckpt}>"]
        print("\n".join(info))

        self.refresh_summary(ckpts)
        if self.dry_run:
            return

        for index, ckpt in enumerate(pending, start=1):
            ckpt_text = f"🚀 Evaluating checkpoint {index}/{len(pending)}"
            print(f"\n{'=' * 80}\n{ckpt_text}\n{'=' * 80}")
            self.checkpoint_manager.load_ckpt_dir = ckpt
            self.checkpoint_manager.load_ckpt(self)
            self.evaluate(ckpt)
            self.refresh_summary(ckpts)

        dist.barrier()
