#!/usr/bin/env bash
#
# convert_text_encoder_to_fsdp_shards.sh
#
# Single-step conversion of T5 and CLIP text encoders to FSDP per-rank shards.
#
# rank=0 loads full weights and broadcasts to all ranks one tensor at a time
# via set_model_state_dict(broadcast_from_rank0=True). FSDP handles sharding.
#
# Usage:
#   convert_text_encoder_to_fsdp_shards.sh <source_path> <output_dir> [nproc] [dtype]
#
# Arguments:
#   source_path   HF FLUX directory (contains text_encoder/, text_encoder_2/).
#   output_dir    Base output dir; t5_fsdp_shards/ and clip_fsdp_shards/ created inside.
#   nproc         Number of GPUs (default: auto-detect).
#   dtype         bfloat16 | float16 | float32 (default: auto-infer from config, fallback float32).
#
# Example:
#   bash scripts/fsdp/convert_text_encoder_to_fsdp_shards.sh \
#       data_and_models/models/flux \
#       data_and_models/models
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/convert_text_encoder_to_fsdp_shards.py"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,23p' "$0" | sed 's/^# *//'; exit 0
fi

if [[ $# -lt 2 ]]; then
  sed -n '2,23p' "$0" | sed 's/^# *//'
  echo "" >&2; echo "Error: need <source_path> and <output_dir>." >&2; exit 1
fi

SOURCE_PATH="$1"
OUTPUT_DIR="$2"
NPROC_ARG="${3:--1}"
DTYPE_ARG="${4:-}"
MASTER_PORT="${MASTER_PORT:-29502}"

if [[ "$NPROC_ARG" == "-1" ]]; then
  NPROC="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [[ "${NPROC}" -eq 0 ]] && { echo "Error: no GPUs found." >&2; exit 1; }
  echo "Auto-detected ${NPROC} GPU(s)."
else
  NPROC="$NPROC_ARG"
fi

DTYPE_LABEL="${DTYPE_ARG:-auto (from config)}"
echo "=== Converting T5 + CLIP to FSDP shards (${NPROC} GPUs, dtype=${DTYPE_LABEL}) ==="
EXTRA_ARGS=()
[[ -n "${DTYPE_ARG}" ]] && EXTRA_ARGS+=(--dtype "${DTYPE_ARG}")
torchrun --nproc_per_node="${NPROC}" \
  --master_port "${MASTER_PORT}" \
  "${PY_SCRIPT}" \
  --source_path "${SOURCE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
