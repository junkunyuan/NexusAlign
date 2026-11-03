"""Experiment logging: rank-aware console and file handlers."""

import os
import sys
import inspect
import logging
import builtins
import warnings

import torch.distributed as dist


_EXPERIMENT_LOGGER_NAME = "Experiment"

ASCTIME_RGB = (80, 220, 120)
EXP_INFO_RGB = (230, 200, 80)
LOCATION_RGB = (120, 200, 230)
RANK_RGB = (230, 80, 80)


class CustomFilter(logging.Filter):
    """
    Logging filter for the distributed context.

    - Attaches the rank, world size, and experiment info to each log record.
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
    """Build the ANSI escape code for a True Color (24-bit RGB) foreground."""
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"


class CustomFormatter(logging.Formatter):
    """
    Rank-aware log formatter.

    - Colors the time, experiment info, rank, and source location fields.
    - Includes the source location when debug_console is enabled.
    """

    def __init__(self, fmt: str, datefmt: str, debug_console: bool = False) -> None:
        super().__init__(fmt, datefmt)
        self.debug_console = debug_console
        self.reset = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format each field with its own color and assemble the line."""
        asctime_str = self.formatTime(record, self.datefmt)
        asc_colored = f"{_rgb_to_code(ASCTIME_RGB)}[{asctime_str}]{self.reset}"

        exp_info_str = getattr(record, "exp_info", "")
        exp_colored = f"{_rgb_to_code(EXP_INFO_RGB)}[{exp_info_str}]{self.reset}"

        message = record.getMessage()
        if self.debug_console:
            filename = getattr(record, "_filename", "")
            func_name = getattr(record, "_func_name", "")
            lineno = getattr(record, "_lineno", "")
            location_str = ""
            if filename and func_name and lineno:
                location_str = f"{filename}({func_name}):{lineno}"
            loc_colored = f"{_rgb_to_code(LOCATION_RGB)}[{location_str}]{self.reset}"

            rank_str = f"rank:{record.rank}/{record.world}"
            rank_colored = f"{_rgb_to_code(RANK_RGB)}[{rank_str}]{self.reset}"

            formatted = f"{asc_colored}{exp_colored}{loc_colored}{rank_colored} {message}"
        else:
            formatted = f"{asc_colored}{exp_colored} {message}"

        return formatted


def init_log(exp_info: str, debug_console: bool = False, exp_dir: str = "logs") -> None:
    """
    Initialize the rank-aware experiment logger.

    Attach the console and file handlers, and replace builtins.print so that print()
    writes through the logger.

    Call this as early as possible after init_dist_env, so that all later print()
    calls share one format.

    The new print sends only print(*args) to the logger. If a call passes file, end,
    sep, or flush, the new print sends the call to the original print, which handles
    these arguments correctly.

    Args:
        exp_info: Experiment identifier shown in each log line.
        debug_console: If True, all ranks log to the console with full error
            messages. Otherwise, only rank 0 logs to the console.
        exp_dir: Directory for the log files. Default: CWD/logs.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()

    if not debug_console:
        warnings.filterwarnings("ignore", category=FutureWarning)
        # ShardedGradScaler.unscale_ passes its [1]-shaped found_inf to the DTensor
        # gradients. DTensor replicates and all-reduces it correctly, but still warns.
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

    custom_filter = CustomFilter(rank, world, exp_info)

    if debug_console or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.addFilter(custom_filter)
        console_handler.setFormatter(
            CustomFormatter(fmt=fmt, datefmt=datefmt, debug_console=debug_console)
        )
        logger.addHandler(console_handler)

    os.makedirs(exp_dir, exist_ok=True)
    file = os.path.join(exp_dir, f"log_rank{rank}.txt")
    file_handler = logging.FileHandler(file, mode="a")
    file_handler.addFilter(custom_filter)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(file_handler)

    original_print = builtins.print

    def log_print(*args, **kwargs):
        if any(k in kwargs for k in ("file", "end", "sep", "flush")):
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
