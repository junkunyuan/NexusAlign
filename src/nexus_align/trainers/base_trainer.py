"""Base trainer: abstract interface and the unified training loop."""

import os
from contextlib import nullcontext
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nexus_align.engine.meter import WindowMeter
from nexus_align.models.base_model import BaseModel
from nexus_align.engine.tracker import TrainingTracker
from nexus_align.engine.checkpoint import CheckpointManager
from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.datasets.dist_dataloader import build_dataloader


class BaseTrainer(ABC):
    """
    Base trainer that owns the generic loop and the order of the training steps.

    - __init__ prepares the dataloader and the step counters from the trainer config.
    - run() handles the epochs, gradient accumulation, logging, checkpoint hooks, and
      EMA updates. On resume, it skips the consumed batches.
    - Subclasses provide the behavior of the model, optimizer, scheduler, and steps.
    """

    def __init__(
        self,
        cfg_trainer: Mapping | None = None,
        train_dataset: Dataset = None,
        model: BaseModel = None,
        algorithm: BaseAlgorithm = None,
        log_dir: str = "logs/temp",
    ) -> None:
        # Get the dataset, model, algorithm, and training config.
        self.train_dataset = train_dataset
        self.model = model
        self.algorithm = algorithm
        for key, value in cfg_trainer.items():
            setattr(self, key, value)

        # Get the dataloader.
        self.train_dataloader = self.get_train_dataloader(train_dataset)

        # Subclasses build the optimizer and the scheduler.
        self.optimizer = None
        self.lr_scheduler = None

        # Build the checkpoint manager.
        self.checkpoint_manager = CheckpointManager(
            save_ckpt_dir=os.path.join(log_dir, "checkpoints"),
            load_ckpt_dir=self.resume_path,
            save_ckpt_every_n_epochs=self.save_ckpt_every_n_epochs,
            save_ckpt_keep_last_n_epochs=self.save_ckpt_keep_last_n_epochs,
            save_ckpt_every_n_steps=self.save_ckpt_every_n_steps,
            save_ckpt_keep_last_n_steps=self.save_ckpt_keep_last_n_steps
        )

        # Components to load and save.
        keys = ["train_state", "model", "ema", "algorithm"]
        self.load_ckpt_keys = (
            keys
            if self.load_ckpt_only_model
            else keys + ["optimizer", "lr_scheduler", "scaler"]
        )
        self.save_ckpt_keys = (
            keys
            if self.save_ckpt_only_model
            else keys + ["optimizer", "lr_scheduler", "scaler"]
        )

        # Other training state.
        self.max_epochs = self.epochs
        self.steps_per_epoch = max(len(self.train_dataloader) // self.grad_accu_step, 1)
        self.max_total_steps = self.max_epochs * self.steps_per_epoch
        self.epoch = 0
        self.step = 0
        self.total_step = 0
        self._metrics = {}
        self._metric_buffer = []  # metrics of the steps since the last log
        self._last_buffered_step = None

        # Get the meters and the tracker.
        self.meters = WindowMeter(
            hardware=self.log_hardware, peak_tflops=self.peak_tflops
        )
        self.meters.register_timing_meters(step_window=self.log_every_n_steps)
        self.meters.set_max_train_state(
            max_epochs=self.max_epochs,
            steps_per_epoch=self.steps_per_epoch,
            max_total_steps=self.max_total_steps,
        )
        self.tracker = TrainingTracker(
            log_dir=log_dir,
            plot_every_n_seconds=self.plot_every_n_seconds,
            log_tensorboard=self.log_tensorboard,
        )

    def get_train_dataloader(self, dataset: Dataset) -> DataLoader:
        """Build the training dataloader from the training dataset."""
        generator = torch.Generator().manual_seed(self.seed)
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=self.shuffle,
            seed=self.seed,
        )
        return build_dataloader(
            dataset=dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            grad_accu_step=self.grad_accu_step,
            generator=generator
        )
    
    @abstractmethod
    def train_mode(self):
        """Switch the trainable modules to training mode."""
        ...

    @abstractmethod
    def zero_grad(self):
        """Clear the accumulated gradients before a global optimization step."""
        ...

    @abstractmethod
    def forward(self, data):
        """Run the forward pass of one micro-batch and return a dict with "loss"."""
        ...

    @abstractmethod
    def backward(self, loss):
        """Backpropagate the already-scaled micro-batch loss."""
        ...

    @abstractmethod
    def no_sync(self):
        """Return a context manager that disables gradient synchronization."""
        ...

    @abstractmethod
    def clip_grad(self):
        """Clip the gradients after accumulation and before the optimizer step."""
        ...

    @abstractmethod
    def optimizer_step(self):
        """Advance the optimizer by one global step."""
        ...

    @abstractmethod
    def lr_scheduler_step(self):
        """Advance the learning-rate scheduler by one global step."""
        ...

    @abstractmethod
    def ema_step(self):
        """Update the EMA weights after the optimizer step."""
        ...

    def update_log(self, epoch_finished: bool = False) -> None:
        """Buffer the metrics of the completed global step, and log every N steps."""
        # Buffer the metrics of each step once. The epoch log repeats the last step.
        if self._last_buffered_step != self.total_step:
            metrics = dict(self._metrics)

            # Get the gradient norm.
            grad_norm = getattr(self, "_grad_norm", None)
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm

            if metrics:
                self._metric_buffer.append(metrics)
            self._last_buffered_step = self.total_step

        should_log = epoch_finished or self.total_step % self.log_every_n_steps == 0
        if not should_log:
            return

        # Average the buffered steps across ranks in one all-reduce. For backend
        # compatibility, it uses SUM and then divides by world_size. Each meter then
        # holds the steps since the last log.
        if self._metric_buffer:
            keys = list(self._metric_buffer[0])
            rows = [[float(m[key]) for key in keys] for m in self._metric_buffer]
            values = torch.as_tensor(rows, dtype=torch.float64, device="cuda")
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= dist.get_world_size()
            for key in keys:
                if key not in self.meters.meters:
                    self.meters.register_meter(key, self.log_every_n_steps)
                self.meters.meters[key].reset()
            for row in values.tolist():
                for key, value in zip(keys, row):
                    self.meters.update(key, value)
            self._metric_buffer = []

        # Track, print, plot, and save the metrics.
        self.tracker.log_all_meters(self.meters, epoch_finished=epoch_finished)

    def state_dict(self) -> dict:
        """Return the trainer runtime state."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "total_step": self.total_step,
            "steps_per_epoch": self.steps_per_epoch
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the trainer runtime state."""
        # Load the epoch.
        self.epoch = int(state_dict["epoch"])

        # Check if steps_per_epoch is the same as in the checkpoint.
        loaded_steps_per_epoch = int(state_dict["steps_per_epoch"])
        data_scale = 1
        if loaded_steps_per_epoch != self.steps_per_epoch:
            print(
                f"⚠️ WARNING: different steps_per_epoch "
                f"checkpoint={loaded_steps_per_epoch}, current={self.steps_per_epoch}"
            )
            data_scale = self.steps_per_epoch / loaded_steps_per_epoch

        # Load the step and the total_step.
        loaded_step = int(state_dict["step"])
        self.step = int(loaded_step * data_scale)
        self.total_step = self.epoch * self.steps_per_epoch + self.step

    def run(self) -> None:
        """Run the training until max_epochs or max_total_steps."""
        # Load the checkpoint and the runtime state.
        self.checkpoint_manager.load_ckpt(self)
        self.meters.update_train_state(self.state_dict())

        # Start TensorBoard and the metric plotter on rank 0.
        self.tracker.open_writer()

        start_epoch = self.epoch
        for _ in range(start_epoch, self.max_epochs):
            reached_max_steps = False

            # Prepare the epoch.
            self.train_mode()
            self.train_dataloader.sampler.set_epoch(self.epoch)
            self.train_dataloader.generator.manual_seed(self.seed + self.epoch)
            data_iter = iter(self.train_dataloader)

            # In a resumed epoch, skip the consumed batches.
            steps_done = self.step if self.epoch == start_epoch else 0
            for _ in range(steps_done * self.grad_accu_step):
                next(data_iter, None)
            if steps_done > 0:
                skipped = steps_done * self.grad_accu_step
                print(f"✅ Skipped {skipped} consumed batches")

            self.meters.start("epoch")
            epoch_finished = False
            for _ in range(steps_done, self.steps_per_epoch):
                self.meters.start("step")

                # 1. Run the forward and backward passes with gradient accumulation.
                self.zero_grad()
                with self.meters.count_flops("step"):
                    for micro_step in range(self.grad_accu_step):
                        data = next(data_iter)

                        # Synchronize the gradients only on the last micro-step.
                        is_final = micro_step == self.grad_accu_step - 1
                        sync_context = nullcontext() if is_final else self.no_sync()
                        with sync_context:
                            forward_results = self.forward(data)
                            self.backward(forward_results["loss"] / self.grad_accu_step)

                # 2. Clip the gradients.
                self.clip_grad()

                # 3. Run the optimizer step.
                self.optimizer_step()

                # 4. Run the LR scheduler step.
                self.lr_scheduler_step()

                # 5. Update EMA.
                self.ema_step()

                # 6. Update the runtime state.
                self.step += 1
                self.total_step += 1
                if self.step >= self.steps_per_epoch:
                    self.epoch += 1
                    self.step = 0
                    epoch_finished = True

                # 7. Save the checkpoint.
                self.checkpoint_manager.save_ckpt(
                    self, epoch_finished=epoch_finished
                )

                # 8. Update the logs.
                self.meters.end("step")
                self.meters.update_train_state(self.state_dict())
                self.update_log()

                if self.total_step >= self.max_total_steps:
                    reached_max_steps = True
                    break

            self.meters.end("epoch")
            self.meters.update_train_state(self.state_dict())
            self.update_log(epoch_finished=True)

            if reached_max_steps:
                break

        self.tracker.close_writer()
        self.meters.stop_hardware_monitoring()

        dist.barrier()
