#!/bin/bash
# ================================================================================
# DiffusionART Training.
# ================================================================================
# Usage:
#     EXP=pickscore bash scripts/reproduce/diffusion_art_train.sh (on every node)
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP=${EXP:-pickscore}  # geneval, ocr, pickscore
echo "🚀 diffusion_art training: EXP=${EXP}"

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
# Prepare the model, prompts, and reward models: download each missing asset.
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
prepare stabilityai/stable-diffusion-3.5-medium
prepare yifan123/flow_grpo
case "$EXP" in
  pickscore)
    prepare yuvalkirstain/PickScore_v1
    prepare laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    ;;
  ocr)
    prepare paddleocr
    ;;
  geneval)
    prepare mask2former
    prepare timm/vit_large_patch14_clip_224.openai
    prepare yifan123/reward-server
    prepare open-mmlab/mmdetection
    ;;
  *) echo "❌ Unknown EXP: ${EXP}"; exit 1 ;;
esac

# --------------------------------------------------------------------------------
# Start the reward server of this node, and stop it when the training ends.
# --------------------------------------------------------------------------------
if [[ "$EXP" == "ocr" || "$EXP" == "geneval" ]]; then
  bash scripts/tools/reward_server.sh ${EXP}
  trap "bash scripts/tools/reward_server.sh ${EXP} stop" EXIT
fi

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
    data=${EXP} \
    algorithm=diffusion_art \
    model=sd3_5_medium \
    trainer=art \
    reward=${EXP} \
    +reproduce=diffusion_art/${EXP} \
    "$@"
