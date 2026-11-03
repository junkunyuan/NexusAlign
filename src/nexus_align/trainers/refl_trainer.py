"""ReFL trainer: online reward feedback learning through a differentiable reward."""

from nexus_align.trainers.grpo_trainer import GRPOTrainer


class ReFLTrainer(GRPOTrainer):
    """
    Generic trainer for reward feedback learning, with the epoch loop of GRPOTrainer.

    - Before the first rollout, the algorithm standardizes the reward with the base
      model on the first calibration_prompts training prompts.
    - Rollouts give the states to train on. The trainer rewards their final images only
      for logging, because the loss backpropagates the reward itself.
    - The trainer trains on every sample and calls ema_step once after each optimizer
      step.

    The algorithm provides the GRPOTrainer interface, and calibrate(prompts, batch_size)
    that sets reward_std. It gets the reward to backpropagate from
    prepare(model, device, pipeline, reward).
    """

    ema_every_batch = False

    def prepare_algorithm(self) -> None:
        self.algorithm.prepare(
            self.model.model, self.model.device, self.model, self.reward
        )

    def sample_epoch(self) -> tuple[dict, dict]:
        # A resumed run loads the calibration with the algorithm state.
        if self.algorithm.reward_std is None:
            if self.total_step > 0:
                raise ValueError("❌ The checkpoint has no reward calibration of ReFL")
            self.model.model.eval()
            prompts = self.train_dataset.texts[:self.algorithm.calibration_prompts]
            self.algorithm.calibrate(prompts, self.sample_batch_size)
        return super().sample_epoch()

    def filter_samples(self, samples: dict) -> dict:
        return samples
