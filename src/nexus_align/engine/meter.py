"""Training meters."""

import os
import time
import threading
import psutil
import pynvml
from collections import deque

import torch


GB = 1024**3


class WindowMeter:
    """A timer and tracker for tracking training information in a window."""

    def __init__(self, hardware: bool = True) -> None:
        self.meters = {
            "exp_start_time": None,
            "total_steps": 0,
            "current_train_steps": 0,
        }

        self.hardware_meters = set()
        self._hardware_monitoring = False
        if hardware:
            self.device_id = torch.cuda.current_device()

            # Get nvml handle
            pynvml.nvmlInit()
            physical_idx = self.device_id
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                visible = [int(x) for x in visible.split(",")]
                physical_idx = visible[self.device_id]
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

            self._monitor_thread = None
            self._stop_monitoring = threading.Event()
            self.hardware_meters = self._init_hardware_meters()
            self.start_hardware_monitoring()

    # ----------------------------------------
    # Meter Utilities
    # ----------------------------------------
    def add_new_meter(
        self, 
        meter: str, 
        window_size: int, 
        decimal: int = 6, 
        report_mean: bool = True
    ) -> None:
        """Add a new meter."""
        self.meters[meter] = {
            "data": deque(maxlen=window_size),
            "mean": "N/A",
            "report_mean": report_mean,
            "decimal": decimal,
        }

    def update_mean(self, meter: str) -> None:
        """Update the mean of a meter."""
        data_list = list(self.meters[meter]["data"])
        self.meters[meter]["mean"] = sum(data_list) / len(data_list)

    def update(self, meter: str, value: object) -> None:
        """Append a value to the meter. Only int and float are accepted."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expect int or float, got {type(value).__name__}. ")
        self.meters[meter]["data"].append(value)
        self.update_mean(meter)

    def latest_exp_info(self) -> dict:
        """Return the latest value for each experiment meter."""
        exp_info = {}
        for k, v in self.meters.items():
            if (
                isinstance(v, dict)
                and k not in self.hardware_meters
                and "step" not in k
                and "epoch" not in k
            ):
                data_list = list(v["data"])
                data = data_list[-1] if len(data_list) > 0 else 0
                exp_info[k] = data
        return exp_info

    # ----------------------------------------
    # Hardware Monitoring
    # ----------------------------------------
    def _init_hardware_meters(self, log_every_n_seconds: int = 3600) -> list[str]:
        """Initialize hardware monitoring meters."""
        hardware_meters = [
            # GPU memory meters (GB)
            "gpu_mem_used", "gpu_mem_peak", "gpu_mem_total",
            # GPU utilization meters (%)
            "gpu_util", "gpu_util_peak",
            # CPU memory meters (GB)
            "cpu_mem_used", "cpu_mem_peak", "cpu_mem_total",
            # CPU utilization meters (%)
            "cpu_util", "cpu_util_peak",
        ]

        for meter in hardware_meters:
            decimal = 0 if "util" in meter else 1
            if "peak" in meter or "total" in meter:
                window_size = 1
            else:
                window_size = log_every_n_seconds
            self.add_new_meter(meter, window_size=window_size, decimal=decimal)

        # Initialize total values
        total_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).total / GB
        total_cpu_memory = psutil.virtual_memory().total / GB
        self.meters["gpu_mem_total"]["data"].append(total_gpu_memory)
        self.meters["cpu_mem_total"]["data"].append(total_cpu_memory)

        return hardware_meters

    def update_peak_status(self, meter: str, value: int | float) -> None:
        """Update the peak status of a meter."""
        if len(self.meters[meter]["data"]) == 0:
            self.update(meter, value)
        else:
            current_peak = max(self.meters[meter]["data"][-1], value)
            self.update(meter, current_peak)

    def _monitor_hardware(self) -> None:
        """Background thread function to monitor hardware."""

        def sample():
            gpu_mem_used = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / GB
            self.update("gpu_mem_used", gpu_mem_used)
            self.update_peak_status("gpu_mem_peak", gpu_mem_used)

            # GPU utilization
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            if gpu_util is not None:
                self.update("gpu_util", gpu_util)
                self.update_peak_status("gpu_util_peak", gpu_util)

            # CPU memory
            cpu_mem_used = psutil.virtual_memory().used / GB
            self.update("cpu_mem_used", cpu_mem_used)
            self.update_peak_status("cpu_mem_peak", cpu_mem_used)

            # CPU utilization
            cpu_util = psutil.cpu_percent(interval=None)
            self.update("cpu_util", cpu_util)
            self.update_peak_status("cpu_util_peak", cpu_util)

        while not self._stop_monitoring.is_set():
            sample()
            if self._stop_monitoring.wait(1):  # sample for every 1 second
                break

    def start_hardware_monitoring(self) -> None:
        """Start hardware monitoring in background thread."""
        if not self._hardware_monitoring:
            self._hardware_monitoring = True
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_hardware, daemon=True
            )
            self._monitor_thread.start()

    def stop_hardware_monitoring(self) -> None:
        """Stop hardware monitoring thread."""
        if self._hardware_monitoring:
            self._stop_monitoring.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1)
            self._hardware_monitoring = False

    # ----------------------------------------
    # Epoch and Step Meters
    # ----------------------------------------
    @property
    def epoch(self) -> int:
        return self.meters["epoch"]["num"]

    @property
    def step(self) -> int:
        return self.meters["step"]["num"]

    @property
    def total_steps(self) -> int:
        return self.meters["total_steps"]

    @property
    def current_train_steps(self) -> int:
        return self.meters["current_train_steps"]

    def add_epoch_step(self, epoch_window: int = 5, step_window: int = 100) -> None:
        """Add epoch and step meters."""
        self.add_new_meter(meter="epoch", window_size=epoch_window, decimal=1)
        self.meters["epoch"]["start_time"] = None
        self.meters["epoch"]["num"] = 0

        self.add_new_meter(meter="step", window_size=step_window, decimal=1)
        self.meters["step"]["start_time"] = None
        self.meters["step"]["num"] = 0

    def update_train_state(self, train_state: dict) -> None:
        """Update the train state."""
        self.meters["epoch"]["num"] = train_state.get("epoch", 0)
        self.meters["step"]["num"] = train_state.get("step", 0)
        self.meters["total_steps"] = train_state.get("total_steps", 0)

    def start(self, meter: str) -> None:
        """Call it at the beginning of every epoch/step."""
        assert meter in ["step", "epoch"], "meter must be 'step' or 'epoch'"
        _time = time.time()
        self.meters[meter]["start_time"] = _time
        if meter == "epoch" and self.meters["exp_start_time"] is None:
            self.meters["exp_start_time"] = _time

    def end(self, meter: str) -> None:
        """Call it at the end of every epoch/step."""
        assert meter in ["step", "epoch"], "meter must be 'step' or 'epoch'"
        self.meters[meter]["num"] += 1
        time_cost = time.time() - self.meters[meter]["start_time"]
        self.update(meter, time_cost)

        if meter == "step":
            self.meters["total_steps"] += 1
            self.meters["current_train_steps"] += 1
        if meter == "epoch":
            self.meters["step"]["num"] = 0
            maxlen = self.meters["step"]["data"].maxlen
            self.meters["step"]["data"] = deque(maxlen=maxlen)

    # ----------------------------------------
    # Print Meter Information
    # ----------------------------------------
    def _get_val(self, meter: str, key: str, unit: str = "") -> str:
        """Get the value to print."""
        meter_data = self.meters[meter][key]
        decimal = self.meters[meter]["decimal"]
        if isinstance(meter_data, deque) and len(meter_data) > 0:
            meter_data = meter_data[-1]
        if isinstance(meter_data, (int, float)):
            return f"{meter_data:.{decimal}f}{unit}"
        return "N/A"

    def info(self, train_info: bool = True, exp_info: bool = True) -> str:
        """Print the information of the meters."""
        get_info = ""

        # Get train state information
        if train_info:
            infos = []
            infos.append(f"epoch: {self.meters['epoch']['num']}")
            infos.append(f"step: {self.meters['step']['num']}")
            infos.append(f"total_steps: {self.meters['total_steps']}")

            for key in ["epoch", "step"]:
                avg_time_cost = self.meters[key]["mean"]
                if isinstance(avg_time_cost, float):
                    avg_time_cost = f"{avg_time_cost:.{self.meters[key]['decimal']}f}s"
                infos.append(f"{key}_avg_time_cost: {avg_time_cost}")

            # Get Hardware information
            if self._hardware_monitoring:
                # GPU memory
                infos.append(
                    f"gpu_mem: {self._get_val('gpu_mem_used', 'data', unit='G')} "
                    f"(mean: {self._get_val('gpu_mem_used', 'mean', unit='G')}, "
                    f"peak: {self._get_val('gpu_mem_peak', 'data', unit='G')}, "
                    f"total: {self._get_val('gpu_mem_total', 'data', unit='G')})"
                )

                # GPU utilization
                infos.append(
                    f"gpu_uti: {self._get_val('gpu_util', 'data', unit='%')} "
                    f"(mean: {self._get_val('gpu_util', 'mean', unit='%')}, "
                    f"peak: {self._get_val('gpu_util_peak', 'data', unit='%')})"
                )

                # CPU memory
                infos.append(
                    f"cpu_mem: {self._get_val('cpu_mem_used', 'data', unit='G')} "
                    f"(mean: {self._get_val('cpu_mem_used', 'mean', unit='G')}, "
                    f"peak: {self._get_val('cpu_mem_peak', 'data', unit='G')}, "
                    f"total: {self._get_val('cpu_mem_total', 'data', unit='G')})"
                )

                # CPU utilization
                infos.append(
                    f"cpu_uti: {self._get_val('cpu_util', 'data', unit='%')} "
                    f"(mean: {self._get_val('cpu_util', 'mean', unit='%')}, "
                    f"peak: {self._get_val('cpu_util_peak', 'data', unit='%')})"
                )

            get_info += "\n📊 " + "  |  ".join(infos)

        # Get experiment information
        if exp_info:
            infos = []
            for k, v in self.meters.items():
                if (
                    isinstance(v, dict)
                    and k not in self.hardware_meters
                    and "step" not in k
                    and "epoch" not in k
                ):
                    data = list(v["data"])
                    decimal = v["decimal"]
                    if len(data) > 0:
                        info = f"{k}: {data[-1]:.{decimal}f}"
                    else:
                        info = f"{k}: N/A"

                    if v.get("report_mean", False):
                        mean_val = v.get("mean", "N/A")
                        try:
                            if hasattr(mean_val, "item"):
                                mean_val = mean_val.item()
                            _mean = f"({mean_val:.{decimal}f})"
                        except (ValueError, TypeError, RuntimeError):
                            _mean = "(N/A)"
                        infos.append(f"{info} {_mean}")
                    else:
                        infos.append(f"{info}")

            get_info += "\n📊 " + "  |  ".join(infos)

        return get_info
