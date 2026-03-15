"""Step 1 of two-step FSDP conversion: random-init FLUX, export placeholder shards."""

import argparse
import json
import os
import textwrap
from functools import partial

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from diffusers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import (
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE_ALIASES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: random-init FLUX, FSDP wrap, export placeholder shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Dtype: --dtype to set; else from transformer/config.json torch_dtype; else float32. No weight file read.
            Run Step 2 afterward to fill weights via CPU.
        """).strip(),
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Shard dtype; else from config or float32",
    )
    return parser.parse_args()


def _dtype_from_config(source_path: str) -> torch.dtype | None:
    path = os.path.join(source_path, "transformer", "config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f).get("torch_dtype")
        if not isinstance(raw, str):
            return None
        s = raw.lower().replace("torch.", "").strip()
        s = DTYPE_ALIASES.get(s, s)
        return DTYPE_MAP.get(s)
    except Exception:
        return None


def _param_init_fn(module: torch.nn.Module, dtype: torch.dtype) -> None:
    for name, param in module.named_parameters(recurse=False):
        if hasattr(param, "is_meta") and param.is_meta:
            t = torch.empty(param.shape, dtype=dtype, device="cpu")
            setattr(module, name, torch.nn.Parameter(t, requires_grad=param.requires_grad))
    for name, buf in module.named_buffers(recurse=False):
        if hasattr(buf, "is_meta") and buf.is_meta:
            setattr(module, name, torch.empty(buf.shape, dtype=dtype, device="cpu"))


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    args = parse_args()
    if args.dtype:
        mp_dtype, dtype_src = DTYPE_MAP[args.dtype], "from --dtype"
    else:
        inferred = _dtype_from_config(args.source_path)
        mp_dtype = inferred if inferred is not None else torch.float32
        dtype_src = "from config" if inferred is not None else "default float32"
    if rank == 0:
        print(f"Using dtype: {mp_dtype} ({dtype_src})")
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    config = FluxTransformer2DModel.load_config(args.source_path, subfolder="transformer")
    with init_empty_weights():
        model = FluxTransformer2DModel.from_config(config, torch_dtype=mp_dtype)
    model = FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={FluxTransformerBlock, FluxSingleTransformerBlock},
        ),
        mixed_precision=MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype),
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
        use_orig_params=True,
        param_init_fn=partial(_param_init_fn, dtype=mp_dtype),
    )

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        local_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )
    out_path = os.path.join(args.output_dir, f"flux_shard-{rank + 1:05d}-of-{world_size:05d}.pt")
    torch.save(local_sd, out_path)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"✅ Step 1 done: {world_size} shards -> {args.output_dir} (dtype={mp_dtype})")


if __name__ == "__main__":
    main()
