"""FSDP2 wrapping utilities."""

from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from nexus_align.utils.dtype import DTYPE_MAP


RESHARD_AFTER_FORWARD = {"full_shard": True, "shard_grad_op": False}


def checkpoint_unless_no_grad(function, *args, **kwargs):
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


def fsdp_wrap(
    module: nn.Module,
    wrap_modules: tuple[type[nn.Module], ...],
    replicate: int = 1,
    shard: int = -1,
    model_name: str = "model",
    strategy: str = "full_shard",
    cpu_offload: bool = False,
    param_dtype: str | None = "bf16",
    reduce_dtype: str = "fp32",
    activation_ckpt: bool = True,
) -> nn.Module:
    """Shard a module in place with FSDP2."""
    if not (isinstance(replicate, int) and isinstance(shard, int)):
        raise ValueError("❌ shard and replicate should be int")
    
    fsdp_kwargs = {}

    # Assgin value to replicate and shard.
    world_size = dist.get_world_size()
    if replicate < 0 and shard < 0:
        shard = min(world_size, 8)
        replicate = world_size // shard
    if shard < 0:
        shard = world_size // replicate
    if replicate < 0:
        replicate = world_size // shard
    if replicate * shard != world_size:
        raise ValueError(
            f"❌ replicate ({replicate}) * shard ({shard}) "
            f"must equal world_size ({world_size})"
        )
    
    # Build mesh.
    if replicate == 1:
        mesh_shape = (shard,)
        mesh_dim_names = ("shard",)
    else:
        mesh_shape = (replicate, shard)
        mesh_dim_names = ("replicate", "shard")
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )
    fsdp_kwargs["mesh"] = mesh

    # FSDP shard strategy.
    if strategy not in RESHARD_AFTER_FORWARD:
        raise ValueError(f"❌ Invalid FSDP strategy: {strategy}")
    fsdp_kwargs["reshard_after_forward"] = RESHARD_AFTER_FORWARD[strategy]

    # FSDP mixed precision; a null param_dtype keeps every parameter's own dtype.
    param_dtype = DTYPE_MAP[param_dtype] if param_dtype else None
    reduce_dtype = DTYPE_MAP[reduce_dtype]
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    # FSDP CPU offload.
    offload_policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    fsdp_kwargs["offload_policy"] = offload_policy

    # Activation wrap.
    if activation_ckpt:
        checkpoint_wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint_unless_no_grad,
        )
        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=lambda m: isinstance(m, wrap_modules),
        )

    # FSDP wrap model.
    for submodule in module.modules():
        if isinstance(submodule, wrap_modules):
            fully_shard(submodule, **fsdp_kwargs)
    fully_shard(module, **fsdp_kwargs)

    total_params = sum(p.numel() for p in module.parameters()) / 1e9  # DTensor.numel is global
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e9
    info = [f"\n📦 Sharded {model_name} with FSDP2:"]
    info += [f"    strategy: {strategy}"]
    info += [f"    fsdp mesh: {tuple(mesh.shape)} (replicate, shard)"]
    info += [f"    wrapped modules: {', '.join(m.__name__ for m in wrap_modules)}"]
    info += [f"    param/reduce dtype: {param_dtype}/{reduce_dtype}"]
    info += [f"    CPU offload: {cpu_offload}"]
    info += [f"    total params: {total_params:.4f} B"]
    info += [f"    trainable params: {trainable_params:.4f} B"]
    print("\n".join(info))

    return module
