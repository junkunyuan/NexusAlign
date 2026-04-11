"""Distributed environment and tools."""

import os

import torch
import torch.distributed as dist


def init_dist_env() -> tuple[int, int, torch.device]:
    """
    Initialize the distributed process group (NCCL backend, env vars).
    Returns (world_size, rank, device).
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
        )

        torch.cuda.set_device(local_rank)

        prefix = "🌐 Distributed environment initialization:  "
        print(f"{prefix}world_size {world_size}  rank {rank}  local_rank {local_rank}")
        dist.barrier()
    else:
        dist_safe_exit(
            exit_code=1, 
            message="❌ Initialize distributed environment failed"
        )

    return world_size, rank, device


def dist_safe_exit(exit_code: int = 0, message: str = "") -> None:
    """Destroy the process group (if initialized) and exit the process."""
    message = f"\n{message}" if message else message
    print(f"🛑 Exiting with code {exit_code}{message}")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("✅ Destroyed process group")
        except Exception as e:
            print(f"⚠️ Warning: Destroy process group failed: {e}")
    os._exit(exit_code)


def all_reduce_tensor(
    x: int | float | list | torch.Tensor,
    op: str = "sum"
) -> int | float | list | torch.Tensor:
    """All-reduce a value across ranks and return in the original type.

    Args:
        x: The value to all-reduce. Supported types: int, float, list, torch.Tensor.
        op: The operation to perform. Supported: "sum", "mean", "max", "min".

    Returns:
        The all-reduced value in the original type.
    """
    ori_type = type(x)
    assert ori_type in (int, float, list, torch.Tensor), f"Invalid type: {ori_type}"

    if isinstance(x, torch.Tensor):
        ori_device = x.device
        tensor = x
        if not x.is_cuda:
            tensor = tensor.cuda()
    else:
        tensor = torch.as_tensor(x, device="cuda")

    reduce_op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.AVG,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    if op not in reduce_op_map:
        raise ValueError(f"❌ Invalid operation: {op}")

    dist.all_reduce(tensor, op=reduce_op_map[op])

    if ori_type is torch.Tensor:
        return tensor.to(ori_device)
    elif ori_type is int:
        return int(tensor.item())
    elif ori_type is float:
        return float(tensor.item())
    elif ori_type is list:
        return tensor.tolist()
