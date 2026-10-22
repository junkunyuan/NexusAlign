"""Metric tracking: scalar logging to Tensorboard and wandb."""

import os
import json

import wandb
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from nexus_align.engine.plotter import MetricPlotter


def init_wandb(**wandb_init_params) -> bool:
    """Initialize wandb on rank 0; init_params are forwarded to wandb.init."""
    if dist.get_rank() == 0 and wandb_init_params["mode"] != "disabled":
        wandb.init(**wandb_init_params)
        wandb.define_metric("total_step")
        wandb.define_metric("epoch")
        wandb.define_metric("total_step/*", step_metric="total_step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        print(f"📌 Wandb initialized ({wandb_init_params['mode']} mode)")
    dist.barrier()

    return wandb.run is not None


class TrainingTracker:
    """
    Rank-0 training metric tracker for tensorboard and wandb.

    - Logs scalars to TensorBoard when a writer is available.
    - Logs scalars to wandb when wandb is initialized.
    - Appends every tracked value to a JSONL result file.
    - Refreshes the metric figures in a background thread.
    """

    def __init__(
        self,
        log_dir: str,
        plot_every_n_seconds: float = 300.0,
        log_tensorboard: bool = False,
    ) -> None:
        self.log_dir = log_dir
        self.log_tensorboard = log_tensorboard
        self.tb_writer = None
        self.save_result_path = os.path.join(log_dir, "results.jsonl")
        self.plotter = MetricPlotter(log_dir, interval=plot_every_n_seconds)

    def open_writer(self) -> None:
        """Open the TensorBoard writer (rank 0 only)."""
        if dist.get_rank() == 0:
            if self.log_tensorboard:
                tb_dir = os.path.join(self.log_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print("📌 TensorBoard used")
            if wandb.run is not None:
                print("📌 Wandb used")
            self.plotter.start()

    def close_writer(self) -> None:
        """Close the TensorBoard writer, the metric plotter, and finish wandb."""
        if dist.get_rank() == 0:
            self.plotter.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if wandb.run is not None:
            wandb.finish()

    def track(
        self,
        name: str,
        value: float,
        total_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Track one metric against the total_step and/or epoch x-axis."""
        x_axes = {"total_step": total_step, "epoch": epoch}
        for axis, x in x_axes.items():
            if x is None:
                continue
            key = f"{axis}/{name}"
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(key, scalar_value=value, global_step=x)
            if wandb.run is not None:
                wandb.log({key: value, axis: x})

    def track_all(
        self,
        metrics: dict,
        total_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Track every metric in the dict against the total_step and/or epoch x-axis."""
        for name, value in metrics.items():
            self.track(name, value, total_step=total_step, epoch=epoch)

    def save_metrics(self, metrics: dict) -> None:
        """Append the latest metrics to a JSONL file."""
        if dist.get_rank() != 0:
            return
        with open(self.save_result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def log_all_meters(self, meters, epoch_finished: bool = False) -> None:
        """Track all meters, print a summary, and append JSONL."""
        metrics = meters.latest_metrics()
        self.track_all(
            metrics,
            epoch=meters.epoch if epoch_finished else None,
            total_step=None if epoch_finished else meters.total_step
        )
        # A step log at step 0 ends an epoch; the epoch log right after prints it.
        if not epoch_finished and meters.step == 0:
            return
        record = {
            "epoch": meters.epoch,
            "step": meters.step,
            "total_step": meters.total_step,
            **meters.all_metrics(),
        }
        if dist.get_rank() == 0:
            print(meters.info(epoch_finished=epoch_finished))
            self.plotter.update(record)
        self.save_metrics(record)
