#!/usr/bin/env bash
#
# convert_flux_to_fsdp_shards.sh
#
# Two-step conversion of FLUX HuggingFace weights to FSDP per-rank shards.
#
# Step 1 (NCCL, GPU): random-init empty FSDP shards; ~5.5GB GPU per rank.
# Step 2 (Gloo, CPU): rank0 loads full weights (~22GB CPU), fills each shard
#                     via dist.send; no GPU involved.
#
# Use the same world_size when loading shards in training.
#
# Usage:
#   convert_flux_to_fsdp_shards.sh <source_path> <output_dir> [nproc]
#
# Arguments:
#   source_path   Path to HF FLUX.1-dev directory (contains transformer/).
#   output_dir    Directory to write shard files.
#   nproc         Number of processes (GPUs). Optional; default -1 = auto-detect
#                 from current machine; pass a number to override.
#
# Example:
#   ./scripts/fsdp/convert_flux_to_fsdp_shards.sh /data/FLUX.1-dev /data/flux_fsdp_shards_ws4
#   ./scripts/fsdp/convert_flux_to_fsdp_shards.sh /data/FLUX.1-dev /data/flux_fsdp_shards_ws8 8
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP1_SCRIPT="${SCRIPT_DIR}/convert_flux_fsdp_step1_random_init.py"
STEP2_SCRIPT="${SCRIPT_DIR}/convert_flux_fsdp_step2_fill_weights.py"

show_help() {
  sed -n '2,29p' "$0" | sed 's/^# *//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

if [[ $# -lt 2 ]]; then
  show_help
  echo "" >&2
  echo "Error: need <source_path> and <output_dir>." >&2
  exit 1
fi

SOURCE_PATH="$1"
OUTPUT_DIR="$2"
NPROC_ARG="${3:--1}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ "$NPROC_ARG" == "-1" ]]; then
  NPROC="$(nvidia-smi -L 2>/dev/null | wc -l)"
  if [[ "${NPROC}" -eq 0 ]]; then
    echo "Error: nproc is -1 (auto) but no GPUs found (nvidia-smi -L returned nothing)." >&2
    exit 1
  fi
  echo "Auto-detected ${NPROC} GPU(s), using nproc=${NPROC}."
else
  NPROC="$NPROC_ARG"
fi

echo "=== Step 1: random-init FSDP shards (NCCL, GPU) ==="
torchrun --nproc_per_node="${NPROC}" \
  --master_port "${MASTER_PORT}" \
  "${STEP1_SCRIPT}" \
  --source_path "${SOURCE_PATH}" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Step 2: fill shards with real weights (Gloo, CPU only) ==="
torchrun --nproc_per_node="${NPROC}" \
  --master_port "${MASTER_PORT}" \
  "${STEP2_SCRIPT}" \
  --source_path "${SOURCE_PATH}" \
  --shard_dir "${OUTPUT_DIR}"
