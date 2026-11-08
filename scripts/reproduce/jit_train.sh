#!/bin/bash
# ================================================================================
# JiT Training.
# ================================================================================
# Usage:
#     EXP=b16 bash scripts/reproduce/jit_train.sh (on every node)
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP=${EXP:-b16}  # b16, l16, h16 (256x256); b32, l32, h32 (512x512)
echo "🚀 jit training: EXP=${EXP}"

# --------------------------------------------------------------------------------
# Prepare the environment.
# --------------------------------------------------------------------------------
export NNODES=${NNODES:-1}  # torchrun variables; set on every node for multi-node runs
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export NCCL_DEBUG=VERSION  # print only the NCCL version

echo "🖥️ NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}" \
  "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

if [ -d ".venv" ]; then
  echo "✅ Environment installed, skipping..."
  source .venv/bin/activate
else
  echo "⏳ Installing environment..."
  pip install uv
  uv venv .venv
  source .venv/bin/activate
  uv sync
fi

# --------------------------------------------------------------------------------
# Prepare ImageNet: download it if it is missing. ImageNet is gated on Hugging
# Face, so set HF_TOKEN first. The first run builds the pixel cache of ImageNet on
# each node, and pMeanFlow uses the same cache.
# --------------------------------------------------------------------------------
prepare() {
  local dst="data_and_model/$1" tmp="data_and_model/.tmp/$1"
  if [[ -e "$dst" ]]; then
    echo "✅ $1 exists, skipping..."
  else
    rm -rf "$tmp"
    python src/nexus_align/utils/download.py "$1" --dir data_and_model/.tmp
    mkdir -p "$(dirname "$dst")"
    mv "$tmp" "$dst"
    echo "💾 Prepared $1"
  fi
}
prepare ILSVRC/imagenet-1k

# --------------------------------------------------------------------------------
# Start the training.
# --------------------------------------------------------------------------------
torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc-per-node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k \
    algorithm=jit \
    model=jit_dit \
    trainer=sft \
    +reproduce=jit/${EXP} \
    "$@"
