"""Training meters: windowed metric tracking and hardware monitoring."""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass

import torch
import psutil
import pynvml

GB = 1024**3

# Hardware groups per device: label, used meter, peak meter, total meter (None if absent), unit.
HARDWARE_SPECS = {
    "gpu": [
        ("mem", "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total", "G"),
        ("uti", "gpu_util", "gpu_util_peak", None, "%"),
    ],
    "cpu": [
        ("mem", "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total", "G"),
        ("uti", "cpu_util", "cpu_util_peak", None, "%"),
    ],
}


def format_time(seconds: float | None) -> str:
    """Format a duration as [Nd ]HH:MM:SS, or "N/A" when it is unknown."""
    if seconds is None:
        return "N/A"
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {text}" if days else text


@dataclass
class Meter:
    """
    A single windowed metric.

    - Keeps a bounded value history and running mean.
    - Uses notation "e" for scientific display.
    - Uses notation "f" for fixed-point display.
    """

    window_size: int
    precision: int = 2
    notation: str = "e"
    report_mean: bool = True

    def __post_init__(self) -> None:
        self.data: deque = deque(maxlen=self.window_size)
        self.mean: float | None = None

    @property
    def latest(self) -> float | None:
        return self.data[-1] if self.data else None

    def update(self, value: int | float) -> None:
        """Append a value and refresh the window mean."""
        self.data.append(value)
        self.mean = sum(self.data) / len(self.data)

    def update_peak(self, value: int | float) -> None:
        """Append the running maximum (windows of size 1 keep a single peak)."""
        self.update(value if not self.data else max(self.data[-1], value))

    def reset(self) -> None:
        """Clear the history while keeping the window size."""
        self.data.clear()
        self.mean = None


class WindowMeter:
    """
    Windowed tracker for training metrics.

    - Tracks user metrics, epoch/step timing, and cumulative counters.
    - Optionally samples GPU/CPU memory and utilization in a background thread.
    - Builds compact log summaries for training progress.
    """

    def __init__(self, hardware: bool = True) -> None:
        self.meters: dict[str, Meter] = {}
        self.hardware_meters: set[str] = set()
        self.timing_meters = {"epoch", "step"}

        # Cumulative counters (plain values, not windowed meters).
        self.epoch = 0
        self.step = 0
        self.total_step = 0
        self.max_epochs = None
        self.steps_per_epoch = None
        self.max_total_steps = None
        self.current_train_steps = 0  # steps in this run
        self.exp_start_time = None
        self._epoch_start = None
        self._step_start = None

        # Hardware monitoring (deps lazily imported; only set up when requested).
        self._hardware_monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._pynvml = None
        self._psutil = None
        self.device_id = None
        self.nvml_handle = None
        if hardware:
            self._setup_hardware()

    # ----------------------------------------
    # Meter Utilities.
    # ----------------------------------------
    def register_meter(
        self,
        meter: str,
        window_size: int,
        precision: int = 2,
        notation: str = "e",
        report_mean: bool = True,
    ) -> None:
        """Register a new windowed meter.

        ``precision`` is the number of mantissa digits in scientific notation
        (``notation``="e", the default for experiment metrics) or the decimal
        places in fixed-point (``notation``="f").
        """
        self.meters[meter] = Meter(window_size, precision, notation, report_mean)

    def update(self, meter: str, value: int | float) -> None:
        """Append a value to a meter. Only int and float are accepted."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"❌ Expected int or float, got {type(value).__name__}")
        self.meters[meter].update(value)

    def update_peak(self, meter: str, value: int | float) -> None:
        """Append the running maximum of a meter."""
        self.meters[meter].update_peak(value)

    def _experiment_meters(self):
        """Yield (name, meter) for user metrics, excluding hardware/timing meters."""
        for name, meter in self.meters.items():
            if name not in self.hardware_meters and name not in self.timing_meters:
                yield name, meter

    def latest_metrics(self) -> dict:
        """Return the latest value of each experiment metric (NaN if empty)."""
        return {
            name: (meter.latest if meter.latest is not None else float("nan"))
            for name, meter in self._experiment_meters()
        }

    def timing_metrics(self) -> dict:
        """Average epoch/step time, elapsed time, and estimated time left, in seconds."""
        metrics = {f"{key}_avg_time": self.meters[key].mean for key in ("epoch", "step")}
        metrics["elapsed_time"] = self.elapsed_time()
        metrics["eta_time"] = self.eta_time()
        return metrics

    def hardware_metrics(self) -> dict:
        """Latest, mean, peak and total of every hardware meter group."""
        if not self._hardware_monitoring:
            return {}
        metrics = {}
        for device, specs in HARDWARE_SPECS.items():
            for label, used, peak, total, _ in specs:
                key = f"{device}_{label}"
                metrics[key] = self.meters[used].latest
                metrics[f"{key}_mean"] = self.meters[used].mean
                metrics[f"{key}_peak"] = self.meters[peak].latest
                if total is not None:
                    metrics[f"{key}_total"] = self.meters[total].latest
        return metrics

    def all_metrics(self) -> dict:
        """Every tracked value: experiment metrics, timing, and hardware usage."""
        return {**self.latest_metrics(), **self.timing_metrics(), **self.hardware_metrics()}

    # ----------------------------------------
    # Hardware Meters.
    # ----------------------------------------
    def _setup_hardware(self) -> None:
        """Resolve the NVML handle, and start sampling."""
        self._psutil = psutil
        self._pynvml = pynvml
        psutil.cpu_percent(interval=None)

        self.device_id = torch.cuda.current_device()
        pynvml.nvmlInit()
        physical_idx = self.device_id
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            physical_idx = [int(x) for x in visible.split(",")][self.device_id]
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

        self.hardware_meters = self._register_hardware_meters()
        self.start_hardware_monitoring()

    def _register_hardware_meters(self) -> set[str]:
        """Register GPU/CPU memory and utilization meters; return their names."""
        log_window = 3600  # seconds of history kept for non-peak meters
        names = [
            "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total",  # GB
            "gpu_util", "gpu_util_peak",                      # %
            "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total",  # GB
            "cpu_util", "cpu_util_peak",                      # %
        ]
        for name in names:
            precision = 0 if "util" in name else 1
            window = 1 if ("peak" in name or "total" in name) else log_window
            self.register_meter(name, window_size=window, precision=precision, notation="f")

        self.update("gpu_mem_total", self._pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).total / GB)
        self.update("cpu_mem_total", self._psutil.virtual_memory().total / GB)
        return set(names)

    def _monitor_hardware(self) -> None:
        """Sample GPU/CPU usage."""
        nvml, ps = self._pynvml, self._psutil
        while not self._stop_monitoring.is_set():
            gpu_mem = nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / GB
            self.update("gpu_mem_used", gpu_mem)
            self.update_peak("gpu_mem_peak", gpu_mem)

            gpu_util = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            if gpu_util is not None:
                self.update("gpu_util", gpu_util)
                self.update_peak("gpu_util_peak", gpu_util)

            cpu_mem = ps.virtual_memory().used / GB
            self.update("cpu_mem_used", cpu_mem)
            self.update_peak("cpu_mem_peak", cpu_mem)

            cpu_util = ps.cpu_percent(interval=None)
            self.update("cpu_util", cpu_util)
            self.update_peak("cpu_util_peak", cpu_util)

            if self._stop_monitoring.wait(1):  # sleep 1s, exit early if stopped
                break

    def start_hardware_monitoring(self) -> None:
        """Start hardware monitoring in a background thread."""
        if not self._hardware_monitoring:
            self._hardware_monitoring = True
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_hardware, daemon=True
            )
            self._monitor_thread.start()

    def stop_hardware_monitoring(self) -> None:
        """Stop the monitoring thread and release NVML."""
        if self._hardware_monitoring:
            self._stop_monitoring.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1)
            self._hardware_monitoring = False
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()

    # ----------------------------------------
    # Epoch and Step Meters.
    # ----------------------------------------
    def register_timing_meters(self, epoch_window: int = 5, step_window: int = 100) -> None:
        """Register the epoch and step timing meters."""
        self.register_meter("epoch", window_size=epoch_window, precision=1, notation="f")
        self.register_meter("step", window_size=step_window, precision=1, notation="f")

    def update_train_state(self, train_state: dict) -> None:
        """Restore the epoch/step/total_step counters (e.g. after resuming)."""
        self.epoch = train_state.get("epoch", 0)
        self.step = train_state.get("step", 0)
        self.total_step = train_state.get("total_step", 0)

    def elapsed_time(self) -> float | None:
        """Seconds since the first epoch of this run started."""
        return None if self.exp_start_time is None else time.time() - self.exp_start_time

    def eta_time(self) -> float | None:
        """Estimated seconds left, from the mean step time and the remaining steps."""
        step_mean = self.meters["step"].mean
        if step_mean is None or self.max_total_steps is None:
            return None
        return max(self.max_total_steps - self.total_step, 0) * step_mean

    def set_max_train_state(
        self, max_epochs: int, steps_per_epoch: int, max_total_steps: int
    ) -> None:
        """Set the training length, shown as the denominator of every counter."""
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_total_steps = max_total_steps

    def start(self, meter: str) -> None:
        """Call it at the beginning of every epoch/step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ Meter must be 'step' or 'epoch'")
        now = time.time()
        if meter == "epoch":
            self._epoch_start = now
            if self.exp_start_time is None:
                self.exp_start_time = now
        else:
            self._step_start = now

    def end(self, meter: str) -> None:
        """Call it at the end of every epoch/step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ meter must be 'step' or 'epoch'")
        if meter == "epoch":
            self.epoch += 1
            self.update("epoch", time.time() - self._epoch_start)
            self.step = 0
        else:
            self.step += 1
            self.total_step += 1
            self.current_train_steps += 1
            self.update("step", time.time() - self._step_start)

    # ----------------------------------------
    # Print Meter Information.
    # ----------------------------------------
    def _fmt(self, meter: str, which: str = "latest", unit: str = "") -> str:
        """Format a meter's latest value or mean using its precision/notation."""
        m = self.meters[meter]
        val = m.latest if which == "latest" else m.mean
        if not isinstance(val, (int, float)):
            return "N/A"
        return f"{val:.{m.precision}{m.notation}}{unit}"

    @staticmethod
    def _fraction(current: int, total: int | None) -> str:
        """Format a counter against its total, or against "-" when it is unknown."""
        return f"{current}/{total if total is not None else '-'}"

    @staticmethod
    def _line(emoji: str, label: str, items: list[str]) -> str:
        """One log line: an emoji, a padded label, and the items of that group."""
        return f"{emoji} {label:<7}  |  " + "  |  ".join(items)

    def _hardware_info(self) -> dict[str, list[str]]:
        """Summary groups (used/mean/peak[/total]) of every hardware device."""
        info = {}
        for device, specs in HARDWARE_SPECS.items():
            groups = []
            for label, used, peak, total, unit in specs:
                group = (
                    f"{label}: {self._fmt(used, 'latest', unit)} "
                    f"(mean: {self._fmt(used, 'mean', unit)}, "
                    f"peak: {self._fmt(peak, 'latest', unit)}"
                )
                if total is not None:
                    group += f", total: {self._fmt(total, 'latest', unit)}"
                groups.append(group + ")")
            info[device] = groups
        return info

    def info(
        self,
        train_info: bool = True,
        show_metrics: bool = True,
        epoch_finished: bool = False,
    ) -> str:
        """Build a human-readable summary string of the meters.

        Layout (one line each):
            🚀 progress  : banner of epoch / step / total_step against their totals
                           (the just-completed epoch when `epoch_finished`, else the
                           epoch in progress)
            📊 metrics   : experiment metrics (loss, ...)
            ⏱️ time      : epoch / step average time cost, elapsed time, and ETA
            🖥️ gpu, cpu  : one line per device, memory and utilization
        """
        lines = []

        if train_info:
            bar = "=" * 80
            if epoch_finished:
                epoch, step = self.epoch, self.steps_per_epoch
            else:
                epoch, step = self.epoch + 1, self.step
            progress = [
                f"epoch {self._fraction(epoch, self.max_epochs)}",
                f"step {self._fraction(step, self.steps_per_epoch)}",
                f"total_step {self._fraction(self.total_step, self.max_total_steps)}",
            ]
            lines += [bar, "🚀 Training  " + "  |  ".join(progress), bar]

        if show_metrics:
            metrics = []
            for name, meter in self._experiment_meters():
                latest = self._fmt(name, "latest")
                if meter.report_mean:
                    metrics.append(f"{name}: {latest} ({self._fmt(name, 'mean')})")
                else:
                    metrics.append(f"{name}: {latest}")
            if metrics:
                lines.append(self._line("📊", "metrics", metrics))

        if train_info:
            timing = [
                f"{key}_avg_time: {self._fmt(key, 'mean', 's')}" for key in ("epoch", "step")
            ]
            timing += [
                f"elapsed: {format_time(self.elapsed_time())}",
                f"eta: {format_time(self.eta_time())}",
            ]
            lines.append(self._line("⏱️", "time", timing))
            for device, groups in self._hardware_info().items():
                lines.append(self._line("🖥️", device, groups))

        return "\n" + "\n".join(lines)
