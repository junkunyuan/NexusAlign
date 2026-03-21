# FSDP Shard Conversion

Converts HuggingFace FLUX weights to FSDP per-rank shards offline.
The `world_size` at conversion must match training.

## convert_flux_to_fsdp_shards.sh

Converts the main transformer. Internally runs two torchrun stages (random-init
via NCCL/GPU, then weight-fill via Gloo/CPU); the script handles both automatically.

```bash
bash scripts/fsdp/convert_flux_to_fsdp_shards.sh <source_path> <output_dir> [nproc]

# auto-detect GPUs
bash scripts/fsdp/convert_flux_to_fsdp_shards.sh \
    data_and_models/models/flux data_and_models/models/flux_fsdp_shards

# explicit nproc
bash scripts/fsdp/convert_flux_to_fsdp_shards.sh \
    data_and_models/models/flux data_and_models/models/flux_fsdp_shards_ws8 8
```

Output: `flux_shard-{rank+1:05d}-of-{world_size:05d}.pt` in `output_dir`.

## convert_text_encoder_to_fsdp_shards.sh

Converts T5 and CLIP text encoders in one step. Rank 0 loads full weights and
broadcasts via `set_model_state_dict(broadcast_from_rank0=True)`.

```bash
bash scripts/fsdp/convert_text_encoder_to_fsdp_shards.sh <source_path> <output_dir> [nproc] [dtype]

bash scripts/fsdp/convert_text_encoder_to_fsdp_shards.sh \
    data_and_models/models/flux data_and_models/models
```

Output: `t5_fsdp_shards/` and `clip_fsdp_shards/` inside `output_dir`.

`dtype`: `bfloat16 | float16 | float32`. Defaults to auto-infer from `config.json`
(falls back to `float32`). Pass `bfloat16` explicitly to avoid double storage size.

## Enable Sharded Loading in Training

```bash
nexus-align train --nproc_per_node=4 model=flux \
    model.fsdp.use_sharded_weights=true \
    model.fsdp.sharded_weights_dir=models/flux_fsdp_shards \
    model.fsdp.t5_fsdp_shards_dir=models/t5_fsdp_shards \
    model.fsdp.clip_fsdp_shards_dir=models/clip_fsdp_shards \
    common.data_and_model_dir=data_and_models
```
