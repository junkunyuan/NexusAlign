"""Distributed: environment init and collective communication helpers."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_dist_env() -> tuple[int, int]:
    """
    Initialize the distributed process group.

    ``torchrun`` is expected for distributed running.
    Environment variables "WORLD_SIZE", "RANK", and "LOCAL_RANK" are expected.
    
    Returns (world_size, rank).
    """
    if all(var in os.environ for var in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]):
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=device,
            timeout=timedelta(minutes=120),
        )

        torch.cuda.set_device(device)

        prefix = "🌐 Distributed environment initialized:  "
        print(f"{prefix}world_size {world_size}  rank {rank}  device {device}")
        dist.barrier()
    else:
        dist_safe_exit(1, message="❌ Initialize distributed environment failed")

    return world_size, rank


def dist_safe_exit(exit_code: int = 0, message: str = "") -> None:
    """Destroy the process group (if initialized) and exit the process."""
    print(f"🛑 Exiting with code {exit_code}\n{message}")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("✅ Destroyed process group")
        except Exception as e:
            print(f"⚠️ Failed to destroy process group: {e}")
    os._exit(exit_code)

