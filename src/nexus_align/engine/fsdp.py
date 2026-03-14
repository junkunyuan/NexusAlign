"""FSDP (Fully Sharded Data Parallel) utilities."""

from functools import partial

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from nexus_align.engine.distributed import all_reduce_tensor

SHARDING_STRATEGY_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}


def _meta_to_cpu_init_fn(m: torch.nn.Module) -> None:
    """Initialize meta tensors to CPU empty tensors for FSDP wrapping with init_empty_weights."""
    for n, p in m.named_parameters(recurse=False):
        if hasattr(p, "is_meta") and p.is_meta:
            t = torch.empty_like(p, device="cpu")
            setattr(m, n, torch.nn.Parameter(t, requires_grad=p.requires_grad))
    for n, b in m.named_buffers(recurse=False):
        if hasattr(b, "is_meta") and b.is_meta:
            setattr(m, n, torch.empty_like(b, device="cpu"))


def convert_scalar_parameters(model: torch.nn.Module) -> None:
    """
    Convert scalar parameters to tensors for FSDP wrapping.
    """
    infos = []
    for name, param in list(model.named_parameters()):
        if param.ndim == 0:
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            scalar_value = param.detach().cpu()
            delattr(module, param_name)
            setattr(module, param_name, scalar_value)
            infos.append({"name": name, "value": scalar_value.item()})

    if infos:
        msg = f"\n⚠️ Converting {len(infos)} scalar parameter(s) to tensor(s) for FSDP wrapping:"
        for info in infos:
            msg += f"\n    param name: {info['name']}, value: {info['value']}"
        print(msg)


def count_fsdp_params(model: torch.nn.Module) -> int:
    """
    Count parameters of FSDP wrapped model.
    """
    count = 0
    for child in model.children():
        if isinstance(child, FSDP):
            count += sum(p.numel() for p in child.parameters())
        else:
            count += count_fsdp_params(child)
    return count


def fsdp_wrap(
    model: torch.nn.Module,
    wrap_modules: list,
    param_dtype: torch.dtype,
    strategy: str,
    cpu_offload: bool,
    # True when loading pre-converted FSDP shards.
    # Model is meta first, then state_dict is loaded.
    from_empty_weights: bool = False,
    model_name: str = "model",
) -> FSDP:
    """Wrap model with torch FSDP (Fully Sharded Data Parallel)."""
    fsdp_kwargs = {}

    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        if recurse:
            return True
        if wrap_modules is None:
            return True
        return any(isinstance(module, m) for m in wrap_modules)

    fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy

    fsdp_kwargs["mixed_precision"] = MixedPrecision(
        param_dtype=param_dtype,
        buffer_dtype=param_dtype,
    )

    if strategy not in SHARDING_STRATEGY_MAP.keys():
        raise ValueError(f"Invalid sharding strategy: {strategy}")
    fsdp_kwargs["sharding_strategy"] = SHARDING_STRATEGY_MAP[strategy]

    # CPU offload
    fsdp_kwargs["cpu_offload"] = CPUOffload(True) if cpu_offload else None

    # Device id
    fsdp_kwargs["device_id"] = torch.cuda.current_device()

    # from_empty_weights: param_init_fn materializes meta tensors, then state_dict is loaded.
    # else: convert_scalar (meta tensors cannot .detach().cpu()).
    if from_empty_weights:
        fsdp_kwargs["param_init_fn"] = _meta_to_cpu_init_fn
    else:
        convert_scalar_parameters(model)

    # Wrap model
    model = FSDP(model, **fsdp_kwargs)

    # Count wrapped parameters
    wrapped_params = 0
    if hasattr(model, "_fsdp_wrapped_module"):
        wrapped_params = count_fsdp_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    wrapped_params = all_reduce_tensor(wrapped_params, op="sum") / 1e9
    total_params = all_reduce_tensor(total_params, op="sum") / 1e9

    wrap_module_names = ["all"]
    if wrap_modules is not None:
        wrap_module_names = [m.__name__ for m in wrap_modules]

    info = [f"\n📦 Wrap {model_name} using FSDP:"]
    info += [f"    Wrapped params: {wrapped_params:.4f} B / {total_params:.4f} B"]
    info += [f"    Strategy: {fsdp_kwargs['sharding_strategy']}"]
    info += [f"    Wrapped modules: {', '.join(wrap_module_names)}"]
    info += [f"    Param dtype: {fsdp_kwargs['mixed_precision'].param_dtype}"]
    info += [f"    CPU Offload: {fsdp_kwargs['cpu_offload']}"]
    print("\n".join(info))

    return model


def activation_wrap(
    model: torch.nn.Module,
    wrap_modules: list,
    model_name: str = "model"
) -> None:
    """Wrap model with activation checkpointing."""
    checkpoint_wrapper_fn = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )

    def check_fn(m):
        if wrap_modules is None:
            return True
        return isinstance(m, tuple(wrap_modules))

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
        check_fn=check_fn,
    )

    print(f"📦 Wrap {model_name} using activation checkpointing")
