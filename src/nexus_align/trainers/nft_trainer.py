"""NFT trainer: online RL on the forward process with a soft-updated sampling policy."""

from torch.utils.data import Dataset

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class NFTTrainer(GRPOTrainer):
    """
    Generic negative-aware fine-tuning trainer, run with the epochs of GRPOTrainer.

    - Every sample is trained, as a zero advantage still gives a loss (r = 0.5).
    - EMA steps once after each optimizer step.
    - After each epoch, the algorithm soft-updates its sampling (old) policy, before the
      checkpoint is saved.
    - Optional stages train one after another, each for its epochs on its own prompts
      with its own rewards of a multi reward, and its own algorithm args; the stage of
      an epoch follows from the epoch, so a resumed run continues in its stage.
    - A new stage starts as DiffusionNFT's resume between stages does: from the EMA
      weights, with a fresh old policy and EMA (equal to them), the optimizer and step
      count kept, and prompts seeded by the epoch within the stage.

    The algorithm provides the GRPOTrainer interface plus update_old_policy(epoch,
    step), and reset_old_policy() for stages.
    """

    def __init__(
        self, *args, stage_datasets: dict[str, Dataset] | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stage_datasets = stage_datasets
        self.stage_index = None
        self.stage_start = 0
        if self.stages is None:
            return
        if sum(stage["epochs"] for stage in self.stages) != self.max_epochs:
            raise ValueError(f"❌ Stage epochs must sum to epochs ({self.max_epochs})")
        # Args a stage overrides return to their defaults in the stages without them.
        self.algorithm_defaults = {
            key: getattr(self.algorithm, key)
            for stage in self.stages for key in stage.get("algorithm", {})
        }

    def set_stage(self) -> None:
        """Switch the prompts, rewards, and algorithm args to this epoch's stage."""
        end = 0
        for index, stage in enumerate(self.stages):
            end += stage["epochs"]
            if self.epoch < end:
                break
        if index == self.stage_index:
            return
        self.stage_index = index
        self.stage_start = end - stage["epochs"]
        if self.epoch == self.stage_start > 0:
            if self.model.ema is not None:
                self.model.load_ema()
            self.algorithm.reset_old_policy()
            print(f"📌 Restarted from the EMA weights at epoch {self.epoch}")
        self.train_dataloader = self.get_train_dataloader(
            self.stage_datasets[stage["data"]]
        )
        self.reward.select(stage["rewards"])
        # Drop the meters of unused rewards, which would log their last values.
        for reward in self.reward.rewards.values():
            for metric in set(reward.metrics) - set(self.reward.metrics):
                self.meters.meters.pop(f"reward_{metric}", None)
        args = {**self.algorithm_defaults, **stage.get("algorithm", {})}
        for key, value in args.items():
            setattr(self.algorithm, key, value)
        print(
            f"📌 Stage {index + 1}/{len(self.stages)}, epochs "
            f"{self.stage_start}-{end}: data {stage['data']}, "
            f"rewards {', '.join(stage['rewards'])}, algorithm args {args}"
        )

    def sampler_epoch(self) -> int:
        return self.epoch - self.stage_start

    def sample_epoch(self) -> tuple[dict, dict]:
        if self.stages is not None:
            self.set_stage()
        return super().sample_epoch()

    def filter_samples(self, samples: dict) -> dict:
        return samples

    def ema_step(self):
        """Step EMA only right after an optimizer step."""
        if self.total_step != self.ema_total_step:
            self.ema_total_step = self.total_step
            super().ema_step()

    def train_epoch(self, samples: dict, rollout_metrics: dict) -> None:
        self.ema_total_step = self.total_step
        super().train_epoch(samples, rollout_metrics)

    def end_epoch(self) -> None:
        self.algorithm.update_old_policy(self.epoch, self.total_step)
