#!/bin/bash
# ================================================================================
# MeanFlow Training.
# ================================================================================
# Usage:
#     EXP=b2 bash scripts/reproduce/meanflow_train.sh (on every node)
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare experiment.
# --------------------------------------------------------------------------------
EXP=${EXP:-b2}  # b2, m2, l2, xl2, xl2_plus
echo "meanflow training: EXP=${EXP}"

# --------------------------------------------------------------------------------
# Prepare environment.
# --------------------------------------------------------------------------------
export NNODES=${NNODES:-1}  # distributed vars; set on every node for multi-node runs
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export NCCL_DEBUG=VERSION  # simple NCCL print

echo NNODES ${NNODES}
echo NODE_RANK ${NODE_RANK}
echo NPROC_PER_NODE ${NPROC_PER_NODE}
echo MASTER_ADDR ${MASTER_ADDR}
echo MASTER_PORT ${MASTER_PORT}

if [ -d ".venv" ]; then
  echo "Environment installed, skipping..."
  source .venv/bin/activate
else
  echo "Installing environment..."
  pip install uv
  uv venv .venv
  source .venv/bin/activate
  uv sync
fi

# --------------------------------------------------------------------------------
# Prepare ImageNet and the VAE: download each missing asset. ImageNet is gated on
# Hugging Face, so set HF_TOKEN first.
# --------------------------------------------------------------------------------
prepare() {
  local dst="data_and_model/$1" tmp="data_and_model/.tmp/$1"
  if [[ -e "$dst" ]]; then
    echo "$1 exists, skipping..."
  else
    rm -rf "$tmp"
    python src/nexus_align/utils/download.py "$1" --dir data_and_model/.tmp
    mkdir -p "$(dirname "$dst")"
    mv "$tmp" "$dst"
    echo "Prepared $1"
  fi
}
prepare ILSVRC/imagenet-1k
prepare stabilityai/sd-vae-ft-mse

# --------------------------------------------------------------------------------
# Start training.
# --------------------------------------------------------------------------------
torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc-per-node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k \
    algorithm=meanflow \
    model=meanflow_dit \
    trainer=sft \
    +reproduce=meanflow/${EXP} \
    "$@"
