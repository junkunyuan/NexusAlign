"""Training logger: TensorBoard, Wandb, and experiment logging."""

import json
import os
import re
import sys
import inspect
import logging
import builtins

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import wandb

from nexus_align.engine.meter import WindowMeter

# Logger name used by init_log; get_experiment_logger() returns it.
_EXPERIMENT_LOGGER_NAME = "Experiment"


def get_experiment_logger() -> logging.Logger:
    """
    Return the experiment logger (name "Experiment").
    After init_log() has been called, this logger has console and file handlers.
    """
    return logging.getLogger(_EXPERIMENT_LOGGER_NAME)


def init_wandb(entity: str, project: str, name: str, wandb_offline: bool) -> bool:
    """
    Initialize Weights & Biases (https://wandb.ai/site).
    Tries online mode first, then falls back to offline. Only rank 0 runs init.

    Args:
        entity: Weights & Biases entity.
        project: Weights & Biases project.
        name: Weights & Biases run name.
        wandb_offline: If True, wandb will be initialized in offline mode.
    Returns:
        True if wandb was initialized successfully, False otherwise.
    """
    wandb_init = False
    if dist.get_rank() == 0:
        init_params = {"entity": entity, "project": project, "name": name}

        if not wandb_offline:
            try:
                wandb.init(mode="online", **init_params)
                wandb_init = True
                print("✅ Wandb initialized (online mode)")
            except Exception as e:
                print(f"⚠️ Wandb online init failed:\n{e}")

        if not wandb_init:
            try:
                wandb.init(mode="offline", **init_params)
                wandb_init = True
                print("✅ Wandb initialized (offline mode)")
            except Exception as e:
                print(f"⚠️ Wandb offline init failed:\n{e}")
                print("⚠️ Training will continue without wandb logging")
    dist.barrier()

    return wandb_init


class RemoveColorFilter(logging.Filter):
    """Remove color codes from log messages."""

    def __init__(self):
        super().__init__()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.ansi_escape.sub("", str(record.msg))
        return True


class CustomFilter(logging.Filter):
    """Add custom information to log records."""

    def __init__(self, rank: int, world: int, exp_info: str) -> None:
        super().__init__()
        self.rank = rank
        self.world = world
        self.exp_info = exp_info

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.world = self.world
        record.exp_info = self.exp_info
        return True


def _rgb_to_code(rgb: tuple[int, int, int]) -> str:
    """Generate ANSI True Color (24-bit RGB) escape code."""
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"


class CustomFormatter(logging.Formatter):
    """Custom formatter to make rank-aware logging with custom colors."""

    def __init__(
        self,
        fmt: str,
        datefmt: str,
        asctime_rgb: tuple[int, int, int],
        rank_rgb: tuple[int, int, int],
        exp_info_rgb: tuple[int, int, int],
        location_rgb: tuple[int, int, int],
        debug_console: bool = False,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.asctime_rgb = asctime_rgb
        self.rank_rgb = rank_rgb
        self.exp_info_rgb = exp_info_rgb
        self.location_rgb = location_rgb
        self.debug_console = debug_console
        self.reset = "\033[0m"

    def format(self, record):
        """Format individual parts with colors."""
        datefmt = getattr(self, "datefmt", "")
        asctime_str = self.formatTime(record, datefmt)
        asc_colored = f"{_rgb_to_code(self.asctime_rgb)}[{asctime_str}]{self.reset}"

        exp_info_str = getattr(record, "exp_info", "")
        exp_colored = f"{_rgb_to_code(self.exp_info_rgb)}[{exp_info_str}]{self.reset}"

        message = record.getMessage()
        if self.debug_console:
            filename = getattr(record, "_filename", "")
            func_name = getattr(record, "_func_name", "")
            lineno = getattr(record, "_lineno", "")
            location_str = ""
            if filename and func_name and lineno:
                location_str = f"{filename}({func_name}):{lineno}"
            loc_colored = f"{_rgb_to_code(self.location_rgb)}[{location_str}]{self.reset}"

            rank_str = f"rank:{record.rank}/{record.world}"
            rank_colored = f"{_rgb_to_code(self.rank_rgb)}[{rank_str}]{self.reset}"

            formatted = f"{asc_colored}{exp_colored}{loc_colored}{rank_colored} {message}"
        else:
            formatted = f"{asc_colored}{exp_colored} {message}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + record.exc_text
        if record.stack_info:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + self.formatStack(record.stack_info)

        return formatted


def init_log(
    exp_info: str,
    debug_console: bool = False,
    exp_dir: str = "logs",
    replace_print: bool = True,
) -> None:
    """
    Initialize the experiment logger (rank-aware), attach console and file handlers,
    and optionally replace builtins.print so that print() goes through this logger.

    Call this as early as possible after init_dist_env() so that all subsequent
    print() and get_experiment_logger().info() output uses the same format. Any
    print() that runs before init_log() (e.g. inside init_dist_env()) is raw stdout.

    Replaced print: only print(*args) is routed to the logger; if file, end, sep,
    or flush are passed, the call is forwarded to the original print for correct semantics.
    Set replace_print=False to leave builtins.print unchanged and use
    get_experiment_logger().info() for structured logging.

    Args:
        exp_info: Experiment identifier shown in each log line.
        debug_console: If True, all ranks log to console with full error messages; else only rank 0.
        exp_dir: Directory for log files. For a single run, pass the run directory
            or a subdirectory (e.g. os.path.join(log_dir, "logs")) so logs sit with
            other run artifacts (config, checkpoints). Default "logs" uses CWD/logs.
        replace_print: If True (default), replace builtins.print with a logger-backed
            implementation. If False, builtins.print is unchanged.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()

    if not debug_console:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

    logger = logging.getLogger(_EXPERIMENT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL + 1)

    fmt = "[%(asctime)s][%(exp_info)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    asctime_rgb = (90, 230, 130)
    exp_info_rgb = (230, 200, 80)
    location_rgb = (120, 200, 230)
    rank_rgb = (230, 80, 80)

    custom_filter = CustomFilter(rank, world, exp_info)

    if debug_console or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.addFilter(custom_filter)
        console_handler.setFormatter(
            CustomFormatter(
                fmt=fmt,
                datefmt=datefmt,
                asctime_rgb=asctime_rgb,
                exp_info_rgb=exp_info_rgb,
                rank_rgb=rank_rgb,
                location_rgb=location_rgb,
                debug_console=debug_console,
            )
        )
        logger.addHandler(console_handler)

    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)
        file = os.path.join(exp_dir, f"log_rank{rank}.txt")
    else:
        file = f"log_rank{rank}.txt"
    file_handler = logging.FileHandler(file, mode="a")
    file_handler.addFilter(custom_filter)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    file_handler.addFilter(RemoveColorFilter())
    logger.addHandler(file_handler)

    if replace_print:
        original_print = builtins.print

        def log_print(*args, old_print=False, **kwargs):
            if old_print or any(k in kwargs for k in ("file", "end", "sep", "flush")):
                original_print(*args, **kwargs)
                return
            message = " ".join(str(a) for a in args)
            caller_frame = inspect.stack()[1]
            filename = caller_frame.filename.split("/")[-1]
            func_name = caller_frame.function
            lineno = caller_frame.lineno
            extra = {"_filename": filename, "_func_name": func_name, "_lineno": lineno}
            logger.info(message, extra=extra)

        builtins.print = log_print


class TrainingLogger:
    """
    Logger for training metrics: TensorBoard and/or Wandb (rank 0 only).
    
    Caller must have initialized wandb externally if wandb_init is True.
    """

    def __init__(self, log_dir: str, wandb_init: bool) -> None:
        self.log_dir = log_dir
        self.wandb_init = wandb_init
        self.tb_writer = None
        self._meters = None  # optional; set via set_meters() for log_all_meters

    def set_meters(self, meters) -> None:
        """Set the meters instance used by log_all_meters (optional)."""
        self._meters = meters

    def open_log(self) -> SummaryWriter | None:
        """Open TensorBoard writer on rank 0."""
        if dist.get_rank() == 0:
            try:
                tb_dir = os.path.join(self.log_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print("✅ TensorBoard used")
            except Exception as e:
                print(f"⚠️ TensorBoard initialization failed: {e}")
            if self.wandb_init:
                print("✅ Wandb used")
            return self.tb_writer
        return None

    def close_log(self) -> None:
        """Close the log writer and optionally finish wandb."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.wandb_init:
            wandb.finish()

    def to_logger(self, name: str, value: float, step: int) -> None:
        """Write a scalar to TensorBoard and/or Wandb (rank 0 only)."""
        if dist.get_rank() != 0:
            return
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)
        if self.wandb_init:
            wandb.log({name: value}, step=step)

    def log_metrics(
        self,
        meters: WindowMeter,
        name: str | None = None,
        value: float | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log metrics to TensorBoard and/or Wandb.
        If name/value are None, log latest window metrics.
        """
        if dist.get_rank() != 0:
            return
        if step is None and meters is not None:
            step = meters.total_steps
        if name is None or value is None:
            if meters is not None:
                exp_info = meters.latest_exp_info()
                for n, v in exp_info.items():
                    self.to_logger(n, v, step)
        else:
            self.to_logger(name, value, step)

    def log_all_meters(self, meters: WindowMeter) -> None:
        """Log all current meters and print a human-readable summary."""
        self.log_metrics(meters)
        if meters is not None and dist.get_rank() == 0:
            print(meters.info())
            self._append_meters_to_jsonl(meters)

    def _append_meters_to_jsonl(self, meters: WindowMeter) -> None:
        """Append the latest meters result to a JSONL file."""
        record = {
            "epoch": meters.epoch, 
            "step": meters.step, 
            "total_steps": meters.total_steps,
            **meters.latest_exp_info()
        }
        jsonl_path = os.path.join(self.log_dir, "results.jsonl")
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Failed to append meters to JSONL: {e}")
