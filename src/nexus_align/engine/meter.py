"""Training meters: windowed metric tracking and hardware monitoring."""

import os
import math
import time
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import psutil
import pynvml
from torch.utils.flop_counter import flop_registry
from torch.utils._python_dispatch import TorchDispatchMode

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

# GPU performance counters that NVML GPM reads on Hopper or newer GPUs: the share of
# SMs that are busy, of cycles in which the Tensor Cores work, and of the peak DRAM
# bandwidth.
GPM_SPECS = [
    ("sm", "gpu_sm_util", "gpu_sm_util_peak", None, "%"),
    ("tensor", "gpu_tensor_util", "gpu_tensor_util_peak", None, "%"),
    ("dram", "gpu_dram_util", "gpu_dram_util_peak", None, "%"),
]
GPM_METRIC_IDS = [
    pynvml.NVML_GPM_METRIC_SM_UTIL,
    pynvml.NVML_GPM_METRIC_ANY_TENSOR_UTIL,
    pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL,
]

# Dense FP16/BF16 TFLOPS of the Tensor Cores of each GPU, for MFU. TF32 runs at half
# of it. The names match in order, so each variant comes before the name that it
# contains.
PEAK_TFLOPS = [
    ("H100 NVL", 835.0),
    ("H100 PCIe", 756.0),
    ("H100", 989.4),
    ("H200 NVL", 835.0),
    ("H200", 989.4),
    ("H800 PCIe", 756.0),
    ("H800", 989.4),
    ("H20", 148.0),
    ("A100", 312.0),
    ("A800", 312.0),
    ("L40S", 362.1),
    ("B200", 2250.0),
]


def format_time(seconds: float | None) -> str:
    """Format a duration as [Nd ]HH:MM:SS, or "N/A" when it is unknown."""
    if seconds is None:
        return "N/A"
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {text}" if days else text


class ModelFlopCounter(TorchDispatchMode):
    """
    Count the model FLOPs of matmul, convolution, and attention ops, for MFU.

    - Splits the FLOPs into FP16/BF16 and TF32 matmuls by the input dtype. FP32
      matmuls run as TF32, which engine/setup.py enables.
    - Skips the forward pass that activation checkpointing recomputes. This pass runs
      inside backward with grad enabled, but backward itself runs without grad.
    - Lets compiled code run compiled. Its matmuls still reach this mode as calls to
      aten kernels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flops = {"16bit": 0, "tf32": 0}

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        formula = flop_registry.get(func._overloadpacket)
        in_backward = torch._C._current_autograd_node() is not None
        if formula is not None and not (in_backward and torch.is_grad_enabled()):
            dtype = next(a.dtype for a in args if isinstance(a, torch.Tensor))
            half = dtype in (torch.float16, torch.bfloat16)
            precision = "16bit" if half else "tf32"
            self.flops[precision] += formula(*args, **kwargs, out_val=out)
        return out


@dataclass
class Meter:
    """
    A single windowed metric.

    - Keeps a bounded value history and a running mean.
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
        """Clear the history, but keep the window size."""
        self.data.clear()
        self.mean = None


class WindowMeter:
    """
    Windowed tracker for training metrics.

    - Tracks the user metrics, the epoch and step timing, and cumulative counters.
    - If hardware is True, samples the GPU and CPU memory and utilization, and the GPU
      performance counters, in a background thread. It also reports MFU from the
      counted FLOPs.
    - Builds compact log summaries of the training progress.
    """

    def __init__(self, hardware: bool = True, peak_tflops: float | None = None) -> None:
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
        self.exp_start_time = None
        self._epoch_start = None
        self._step_start = None

        # Hardware monitoring, which starts only if hardware is True.
        self._hardware_monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self.device_id = None
        self.nvml_handle = None
        self.hardware_specs = {}
        self._gpm_samples = None

        # MFU: the model FLOPs of one epoch or step on this rank, over the peak.
        self.peak_tflops = peak_tflops  # None looks it up by GPU name
        self.flops = None  # (timing meter, {precision: FLOPs})
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
        """
        Register a new windowed meter.

        ``precision`` is the number of mantissa digits in scientific notation
        (``notation``="e", the default for experiment metrics) or the decimal
        places in fixed-point (``notation``="f").
        """
        self.meters[meter] = Meter(window_size, precision, notation, report_mean)

    def update(self, meter: str, value: int | float) -> None:
        """Append a value to a meter. The value must be an int or a float."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"❌ Expected int or float, got {type(value).__name__}")
        self.meters[meter].update(value)

    def update_peak(self, meter: str, value: int | float) -> None:
        """Append the running maximum of a meter."""
        self.meters[meter].update_peak(value)

    def _experiment_meters(self):
        """Yield (name, meter) for user metrics, except hardware and timing meters."""
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
        for device, specs in self.hardware_specs.items():
            for label, used, peak, total, _ in specs:
                key = f"{device}_{label}"
                metrics[key] = self.meters[used].latest
                metrics[f"{key}_mean"] = self.meters[used].mean
                metrics[f"{key}_peak"] = self.meters[peak].latest
                if total is not None:
                    metrics[f"{key}_total"] = self.meters[total].latest
        metrics["gpu_mfu"] = self.mfu()
        return metrics

    def all_metrics(self) -> dict:
        """Every tracked value: experiment metrics, timing, and hardware usage."""
        return {**self.latest_metrics(), **self.timing_metrics(), **self.hardware_metrics()}

    # ----------------------------------------
    # Hardware Meters.
    # ----------------------------------------
    def _setup_hardware(self) -> None:
        """Get the NVML handle and start the sampling."""
        psutil.cpu_percent(interval=None)

        self.device_id = torch.cuda.current_device()
        pynvml.nvmlInit()
        physical_idx = self.device_id
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            physical_idx = [int(x) for x in visible.split(",")][self.device_id]
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

        if self.peak_tflops is None:
            name = torch.cuda.get_device_name(self.device_id)
            self.peak_tflops = next((v for k, v in PEAK_TFLOPS if k in name), None)
            if self.peak_tflops is None:
                print(f"⚠️ Unknown peak TFLOPS of <{name}>; set trainer.peak_tflops")

        self._gpm_samples = self._setup_gpm()
        self.hardware_specs = {
            "gpu": HARDWARE_SPECS["gpu"] + (GPM_SPECS if self._gpm_samples else []),
            "cpu": HARDWARE_SPECS["cpu"],
        }
        self.hardware_meters = self._register_hardware_meters()
        self.start_hardware_monitoring()

    def _register_hardware_meters(self) -> set[str]:
        """Register GPU/CPU memory and utilization meters, and return their names."""
        log_window = 3600  # seconds of history kept for non-peak meters
        names = [
            "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total",  # GB
            "gpu_util", "gpu_util_peak",                      # %
            "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total",  # GB
            "cpu_util", "cpu_util_peak",                      # %
        ]
        for _, used, peak, _, _ in GPM_SPECS if self._gpm_samples else []:
            names += [used, peak]                             # %
        for name in names:
            precision = 0 if "util" in name else 1
            window = 1 if ("peak" in name or "total" in name) else log_window
            self.register_meter(name, window_size=window, precision=precision, notation="f")

        gpu_mem_total = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).total
        self.update("gpu_mem_total", gpu_mem_total / GB)
        self.update("cpu_mem_total", psutil.virtual_memory().total / GB)
        return set(names)

    def _setup_gpm(self) -> list | None:
        """Allocate two GPM samples and take the first, or return None without GPM."""
        try:
            if not pynvml.nvmlGpmQueryDeviceSupport(self.nvml_handle).isSupportedDevice:
                return None
            samples = [pynvml.nvmlGpmSampleAlloc() for _ in range(2)]
            pynvml.nvmlGpmSampleGet(self.nvml_handle, samples[0])
        except pynvml.NVMLError:
            return None
        return samples

    def _read_gpm(self) -> list[float]:
        """Read the GPM metrics over the interval since the previous sample, in %."""
        previous, current = self._gpm_samples
        pynvml.nvmlGpmSampleGet(self.nvml_handle, current)
        request = pynvml.c_nvmlGpmMetricsGet_t()
        request.version = pynvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(GPM_METRIC_IDS)
        request.sample1, request.sample2 = previous, current
        for i, metric_id in enumerate(GPM_METRIC_IDS):
            request.metrics[i].metricId = metric_id
        pynvml.nvmlGpmMetricsGet(request)
        self._gpm_samples = [current, previous]
        return [request.metrics[i].value for i in range(len(GPM_METRIC_IDS))]

    def _monitor_hardware(self) -> None:
        """Sample GPU/CPU usage."""
        while not self._stop_monitoring.is_set():
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / GB
            self.update("gpu_mem_used", gpu_mem)
            self.update_peak("gpu_mem_peak", gpu_mem)

            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            self.update("gpu_util", gpu_util)
            self.update_peak("gpu_util_peak", gpu_util)

            if self._gpm_samples is not None:
                try:
                    values = self._read_gpm()
                except pynvml.NVMLError:  # for example, a profiler holds the counters
                    values = []
                for (_, used, peak, _, _), value in zip(GPM_SPECS, values):
                    if math.isfinite(value):  # NaN for a too-short interval
                        self.update(used, value)
                        self.update_peak(peak, value)

            cpu_mem = psutil.virtual_memory().used / GB
            self.update("cpu_mem_used", cpu_mem)
            self.update_peak("cpu_mem_peak", cpu_mem)

            cpu_util = psutil.cpu_percent(interval=None)
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
            for sample in self._gpm_samples or []:
                pynvml.nvmlGpmSampleFree(sample)
            pynvml.nvmlShutdown()

    @contextmanager
    def count_flops(self, meter: str):
        """Count the model FLOPs of the first epoch or step it wraps, for MFU."""
        if self.flops is not None or not self._hardware_monitoring:
            yield
            return
        with ModelFlopCounter() as counter:
            yield
        self.flops = (meter, counter.flops)

    def mfu(self) -> float | None:
        """Model FLOPs utilization in %: the model FLOPs per second over the peak."""
        if self.flops is None or self.peak_tflops is None:
            return None
        meter, flops = self.flops
        seconds = self.meters[meter].mean
        if seconds is None:
            return None
        # The time that the FLOPs take at the peak, with TF32 at half the peak.
        peak_seconds = (flops["16bit"] + 2 * flops["tf32"]) / (self.peak_tflops * 1e12)
        return 100 * peak_seconds / seconds

    # ----------------------------------------
    # Epoch and Step Meters.
    # ----------------------------------------
    def register_timing_meters(
        self, epoch_window: int = 10, step_window: int = 100
    ) -> None:
        """Register the epoch and step timing meters."""
        self.register_meter("epoch", window_size=epoch_window, precision=1, notation="f")
        self.register_meter("step", window_size=step_window, precision=1, notation="f")

    def update_train_state(self, train_state: dict) -> None:
        """Restore the epoch, step, and total_step counters, for example on resume."""
        self.epoch = train_state["epoch"]
        self.step = train_state["step"]
        self.total_step = train_state["total_step"]

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
        """Call it at the beginning of every epoch or step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ Meter must be 'epoch' or 'step'")
        now = time.time()
        if meter == "epoch":
            self._epoch_start = now
            if self.exp_start_time is None:
                self.exp_start_time = now
        else:
            self._step_start = now

    def end(self, meter: str) -> None:
        """Call it at the end of every epoch or step."""
        if meter not in self.timing_meters:
            raise ValueError("❌ meter must be 'epoch' or 'step'")
        if meter == "epoch":
            self.epoch += 1
            self.update("epoch", time.time() - self._epoch_start)
            self.step = 0
        else:
            self.step += 1
            self.total_step += 1
            self.update("step", time.time() - self._step_start)

    # ----------------------------------------
    # Print Meter Information.
    # ----------------------------------------
    def _fmt(self, meter: str, which: str = "latest", unit: str = "") -> str:
        """Format a meter's latest value or mean with its precision and notation."""
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
        if not self._hardware_monitoring:
            return {}
        info = {}
        for device, specs in self.hardware_specs.items():
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
        if info:
            mfu = self.mfu()
            info["gpu"].append(f"mfu: {'N/A' if mfu is None else f'{mfu:.1f}%'}")
        return info

    def info(self, epoch_finished: bool = False) -> str:
        """
        Build a human-readable summary string of the meters.

        Layout (one line each):
            🚀 Training  : banner of epoch / step / total_step against their totals
                           (the completed epoch if `epoch_finished`, otherwise the
                           epoch in progress)
            📊 metrics   : experiment metrics (loss, ...)
            ⏱️ time      : average time of an epoch and a step, elapsed time, and ETA
            🖥️ gpu, cpu  : one line per device, memory and utilization, plus the GPU
                           SM, Tensor Core, and DRAM utilization and MFU
        """
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
        lines = [bar, "🚀 Training  " + "  |  ".join(progress), bar]

        metrics = []
        for name, meter in self._experiment_meters():
            latest = self._fmt(name, "latest")
            if meter.report_mean:
                metrics.append(f"{name}: {latest} ({self._fmt(name, 'mean')})")
            else:
                metrics.append(f"{name}: {latest}")
        if metrics:
            lines.append(self._line("📊", "metrics", metrics))

        timing = [
            f"{key}_avg_time: {self._fmt(key, 'mean', 's')}"
            for key in ("epoch", "step")
        ]
        timing += [
            f"elapsed: {format_time(self.elapsed_time())}",
            f"eta: {format_time(self.eta_time())}",
        ]
        lines.append(self._line("⏱️", "time", timing))
        for device, groups in self._hardware_info().items():
            lines.append(self._line("🖥️", device, groups))

        return "\n" + "\n".join(lines)
