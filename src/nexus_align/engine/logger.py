"""Experiment logging: rank-aware console and file handlers."""

import os
import re
import sys
import inspect
import logging
import builtins
import warnings

import torch.distributed as dist


_EXPERIMENT_LOGGER_NAME = "Experiment"


class RemoveColorFilter(logging.Filter):
    """
    Logging filter for file output.

    - Strips ANSI color codes from log messages.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.ansi_escape.sub("", str(record.msg))
        return True


class CustomFilter(logging.Filter):
    """
    Logging filter for distributed context.

    - Attaches rank, world size, and experiment info to each log record.
    """

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
    """Build an ANSI True Color (24-bit RGB) foreground escape code."""
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"


class CustomFormatter(logging.Formatter):
    """
    Rank-aware log formatter.

    - Colors time, experiment info, rank, and source location fields.
    - Includes source location when debug_console is enabled.
    """

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

    def format(self, record: logging.LogRecord) -> str:
        """Format each field with its own color and assemble the line."""
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

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if formatted[-1:] != "\n":
                formatted += "\n"
            formatted += record.exc_text
        if record.stack_info:
            if formatted[-1:] != "\n":
                formatted += "\n"
            formatted += self.formatStack(record.stack_info)

        return formatted


def init_log(
    exp_info: str,
    debug_console: bool = False,
    exp_dir: str = "logs",
    replace_print: bool = True,
) -> None:
    """
    Initialize the rank-aware experiment logger.

    Attaches console and file handlers, and optionally replaces builtins.print
    so that print() is routed through the logger.

    When to call this: As early as possible after initialize distributed environment 
    so that subsequent print() and get_experiment_logger().info() share one format.

    The replaced print only routes print(*args) to the logger; if file, end,
    sep, or flush are passed, the call is forwarded to the original print for
    correct semantics. Set replace_print=False to leave builtins.print
    unchanged and use get_experiment_logger().info() for structured logging.

    Args:
        exp_info: Experiment identifier shown in each log line.
        debug_console: If True, all ranks log to console with full error
            messages; otherwise only rank 0 logs to console.
        exp_dir: Directory for log files. Default: CWD/logs.
        replace_print: If True (default), replace builtins.print with a
            logger-backed implementation; otherwise builtins.print is unchanged.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()

    if not debug_console:
        warnings.filterwarnings("ignore", category=FutureWarning)
        # ShardedGradScaler.unscale_ passes its [1]-shaped found_inf to DTensor grads,
        # which DTensor replicates and all-reduces correctly but still warns about.
        warnings.filterwarnings(
            "ignore", message="Found a non-scalar tensor with numel=1", category=UserWarning
        )

    logger = logging.getLogger(_EXPERIMENT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL + 1)

    fmt = "[%(asctime)s][%(exp_info)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    asctime_rgb = (80, 220, 120)
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
            extra = {
                "_filename": caller_frame.filename.split("/")[-1],
                "_func_name": caller_frame.function,
                "_lineno": caller_frame.lineno,
            }
            logger.info(message, extra=extra)

        builtins.print = log_print
