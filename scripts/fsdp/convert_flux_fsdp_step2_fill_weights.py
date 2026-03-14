"""Step 2 of two-step FSDP conversion: fill placeholder shards with real pretrained weights."""

import argparse
import os
import textwrap

import torch
import torch.distributed as dist
from diffusers import FluxTransformer2DModel
from torch.distributed._shard.sharded_tensor import ShardedTensor
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: rank0 loads full pretrained weights (CPU only), "
        "slices and sends each rank its portion via gloo, overwrites placeholder shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Usage:
              torchrun --nproc_per_node=4 scripts/fsdp/convert_flux_fsdp_step2_fill_weights.py \\
                  --source_path data_and_models/models/flux \\
                  --shard_dir   data_and_models/models/flux_fsdp_shards

            Memory: rank0 ~22GB CPU (full pretrained) + shard size; other ranks ~shard size only.
            No GPU required (gloo backend, pure CPU communication).
        """).strip(),
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to HF FLUX.1-dev directory (loads transformer/ pretrained weights)",
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        required=True,
        help="Directory containing placeholder shards from Step 1",
    )
    return parser.parse_args()


def _parse_rank(placement_str: str) -> int:
    s = str(placement_str)
    i = s.find("rank:")
    if i < 0:
        return -1
    j = s.find("/", i)
    return int(s[i + 5 : j] if j > i else s[i + 5 :])


def _slice_tensor(t: torch.Tensor, offsets: list, sizes: list) -> torch.Tensor:
    sl = [slice(o, o + sz) for o, sz in zip(offsets, sizes)]
    return t[tuple(sl)]


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        print("WORLD_SIZE must be > 1")
        return

    args = parse_args()

    dist.init_process_group(backend="gloo")

    try:
        shard_path = os.path.join(
            args.shard_dir, f"flux_shard-{rank + 1:05d}-of-{world_size:05d}.pt"
        )
        if not os.path.exists(shard_path):
            raise FileNotFoundError(shard_path)

        sd = torch.load(shard_path, map_location="cpu", weights_only=False)

        if rank == 0:
            print("Start covering shards", flush=True)
            print(
                f"Loading pretrained on rank0 from {args.source_path}/transformer", flush=True
            )
            model = FluxTransformer2DModel.from_pretrained(
                args.source_path, subfolder="transformer"
            )
            print("Loaded pretrained on rank0", flush=True)
            params = dict(model.named_parameters())
            keys = [k for k, v in sd.items() if isinstance(v, ShardedTensor) and k in params]
            pbar = tqdm(total=len(keys))
        else:
            print("Start covering shards", flush=True)
            print("Waiting for rank0 to broadcast keys", flush=True)
            params = None
            keys = None
            pbar = None

        obj = [keys]
        dist.broadcast_object_list(obj, src=0)
        keys = obj[0]
        dist.barrier()

        if rank == 0:
            for k in keys:
                st = sd[k]
                meta = st.metadata()
                gshape = tuple(meta.size)
                ft = params[k].data
                if tuple(ft.shape) != gshape:
                    raise RuntimeError(f"shape mismatch: {k}")
                tdtype = meta.tensor_properties.dtype
                lshards = st.local_shards()
                lshards = sorted(lshards, key=lambda s: tuple(s.metadata.shard_offsets))
                for ls in lshards:
                    off = ls.metadata.shard_offsets
                    sz = ls.metadata.shard_sizes
                    sl = _slice_tensor(ft, off, sz).to(dtype=ls.tensor.dtype)
                    with torch.no_grad():
                        ls.tensor.copy_(sl)
                for r in range(1, world_size):
                    sm = [m for m in meta.shards_metadata if _parse_rank(m.placement) == r]
                    sm = sorted(sm, key=lambda m: tuple(m.shard_offsets))
                    for m in sm:
                        off = m.shard_offsets
                        sz = m.shard_sizes
                        sl = _slice_tensor(ft, off, sz).to(dtype=tdtype).contiguous()
                        dist.send(sl, dst=r)
                dist.barrier()
                pbar.set_postfix_str(str(tuple(ft.shape)))
                pbar.update(1)
            pbar.close()
        else:
            for k in keys:
                v = sd.get(k, None)
                if not isinstance(v, ShardedTensor):
                    raise RuntimeError(f"missing or invalid sharded tensor: {k}")
                st = v
                lshards = st.local_shards()
                if not lshards:
                    dist.barrier()
                    continue
                lshards = sorted(lshards, key=lambda s: tuple(s.metadata.shard_offsets))
                for ls in lshards:
                    buf = torch.empty_like(ls.tensor)
                    dist.recv(buf, src=0)
                    with torch.no_grad():
                        ls.tensor.copy_(buf)
                dist.barrier()

        torch.save(sd, shard_path)
        dist.barrier()
        print("Done covering shards")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
