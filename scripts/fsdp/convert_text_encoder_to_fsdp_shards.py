"""Convert T5 and CLIP text encoders to FSDP per-rank shards (single-step)."""

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import CLIPTextModel, T5EncoderModel
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
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
        description="Convert T5 and CLIP text encoders to FSDP shards (single-step)."
    )
    parser.add_argument(
        "--source_path", type=str, required=True,
        help="Path to HF FLUX directory (contains text_encoder/, text_encoder_2/).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base output dir; t5_fsdp_shards/ and clip_fsdp_shards/ created inside.",
    )
    parser.add_argument(
        "--dtype", type=str, default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Shard dtype; else inferred from config or float32",
    )
    return parser.parse_args()


def _dtype_from_config(source_path: str, subfolder: str) -> torch.dtype | None:
    import json

    path = os.path.join(source_path, subfolder, "config.json")
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
            setattr(
                module, name,
                torch.nn.Parameter(
                    torch.empty(param.shape, dtype=dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                ),
            )
    for name, buf in module.named_buffers(recurse=False):
        if hasattr(buf, "is_meta") and buf.is_meta:
            setattr(module, name, torch.empty(buf.shape, dtype=dtype, device="cpu"))


def _convert_scalar_parameters(model: torch.nn.Module) -> None:
    for name, param in list(model.named_parameters()):
        if param.ndim == 0:
            parts = name.split(".")
            m = model
            for part in parts[:-1]:
                m = getattr(m, part)
            scalar = param.detach().cpu()
            delattr(m, parts[-1])
            setattr(m, parts[-1], scalar)


def _fsdp_wrap(
    model: torch.nn.Module,
    wrap_cls: type,
    dtype: torch.dtype,
    local_rank: int,
) -> FSDP:
    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    return FSDP(
        model,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls={wrap_cls}
        ),
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
        device_id=local_rank,
        use_orig_params=True,
        param_init_fn=partial(_param_init_fn, dtype=dtype),
    )


def _convert_encoder(
    model_cls: type,
    wrap_cls: type,
    source_path: str,
    subfolder: str,
    dtype: torch.dtype,
    local_rank: int,
    rank: int,
    world_size: int,
    output_dir: str,
    shard_prefix: str,
) -> None:
    if rank == 0:
        print(f"⏳ Converting {subfolder} ({dtype}) -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    # All ranks: build empty model structure via meta device
    config = model_cls.config_class.from_pretrained(source_path, subfolder=subfolder)
    with init_empty_weights():
        model = model_cls(config)
    model.eval()
    model.requires_grad_(False)
    _convert_scalar_parameters(model)

    # FSDP wrap: param_init_fn materializes meta tensors to CPU empty tensors
    model = _fsdp_wrap(model, wrap_cls, dtype, local_rank)

    # rank=0: load full weights; others: empty dict (broadcast fills them)
    full_sd = {}
    if rank == 0:
        full_sd = model_cls.from_pretrained(
            source_path, subfolder=subfolder, torch_dtype=dtype, low_cpu_mem_usage=True
        ).state_dict()
        if "shared.weight" in full_sd and "encoder.embed_tokens.weight" not in full_sd:
            full_sd["encoder.embed_tokens.weight"] = full_sd["shared.weight"]

    # Broadcast from rank=0 -> FSDP scatters each rank's portion
    # Internally broadcasts tensors one-by-one to avoid OOM
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        set_model_state_dict(
            model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True, strict=False
            ),
        )

    # Each rank extracts and saves its local shard
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        local_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )
    shard_path = os.path.join(
        output_dir, f"{shard_prefix}-{rank + 1:05d}-of-{world_size:05d}.pt"
    )
    torch.save(local_sd, shard_path)

    if rank == 0:
        print(f"✅ {subfolder}: {world_size} shards -> {output_dir}")

    del model, full_sd, local_sd
    torch.cuda.empty_cache()
    dist.barrier()


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    args = parse_args()
    if args.dtype:
        dtype = DTYPE_MAP[args.dtype]
        dtype_src = "from --dtype"
    else:
        inferred = _dtype_from_config(args.source_path, "text_encoder_2")
        dtype = inferred if inferred is not None else torch.float32
        dtype_src = "from config" if inferred is not None else "default float32"
    if rank == 0:
        print(f"Using dtype: {dtype} ({dtype_src})")

    _convert_encoder(
        model_cls=T5EncoderModel,
        wrap_cls=T5Block,
        source_path=args.source_path,
        subfolder="text_encoder_2",
        dtype=dtype,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        output_dir=os.path.join(args.output_dir, "t5_fsdp_shards"),
        shard_prefix="t5_shard",
    )

    _convert_encoder(
        model_cls=CLIPTextModel,
        wrap_cls=CLIPEncoderLayer,
        source_path=args.source_path,
        subfolder="text_encoder",
        dtype=dtype,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        output_dir=os.path.join(args.output_dir, "clip_fsdp_shards"),
        shard_prefix="clip_shard",
    )

    dist.destroy_process_group()
    if rank == 0:
        print(f"✅ All done. Shards in {args.output_dir}/{{t5,clip}}_fsdp_shards/")


if __name__ == "__main__":
    main()
