"""Run provenance: code, software, and hardware facts that make a run reproducible."""

import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    """Run a git command in the project dir; return its stdout or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str | None:
    """Read the CPU model name from /proc/cpuinfo (Linux only)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def collect_provenance() -> dict:
    """
    Collect non-private facts about the run: code version, software versions, hardware.

    Hostname, IP addresses, user names, and environment variables are deliberately excluded.
    """
    dirty = _git("status", "--porcelain")
    git = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if dirty is None else bool(dirty),
    }
    software = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()) if torch.cuda.is_available() else None,
    }
    hardware = {
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_gb": None,
        "gpu_capability": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware.update({
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
            "gpu_capability": f"{props.major}.{props.minor}",
        })
    return {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "software": software,
        "hardware": hardware,
    }
