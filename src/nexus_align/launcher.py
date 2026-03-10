"""Distributed training launcher using torchrun."""

import os
import importlib.util
import subprocess
import sys

# Default env vars set before running the script
_LAUNCH_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "HYDRA_FULL_ERROR": "1"
}


def _resolve_script_path(script: str) -> str:
    """Resolve module name to .py file path for torchrun."""
    if script.endswith(".py"):
        return script
    spec = importlib.util.find_spec(script)
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot resolve script: {script}")
    return spec.origin


def launch(
    script: str = "nexus_align.cli.train",
    script_args: list[str] | None = None,
    *,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    rdzv_id: str = "nexus_align",
) -> int:
    """
    Launch distributed training via `torch.distributed.run` (a.k.a. `torchrun`).

    Args:
        script: Python module to run (e.g. "nexus_align.cli.train") or path to .py file.
        script_args: Additional arguments passed to the script.
        nproc_per_node: Number of GPUs per node.
        nnodes: Total number of nodes.
        node_rank: Rank of the current node (0-indexed).
        master_addr: Master node address (IP or hostname).
        master_port: Master port for rendezvous.
        rdzv_id: Unique job ID for rendezvous (isolates concurrent runs).

    Returns:
        Exit code of the launched process.
    """
    script_args = script_args or []
    script_path = _resolve_script_path(script)
    rdzv_endpoint = f"{master_addr}:{master_port}"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes", str(nnodes),
        "--nproc_per_node", str(nproc_per_node),
        "--node_rank", str(node_rank),
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", rdzv_endpoint,
        "--rdzv_id", rdzv_id,
        script_path,
        *script_args,
    ]

    env = os.environ.copy()
    env.update(_LAUNCH_ENV_DEFAULTS)

    return subprocess.run(cmd, env=env).returncode
