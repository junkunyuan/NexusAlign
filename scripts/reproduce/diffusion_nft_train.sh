#!/bin/bash
# ================================================================================
# DiffusionNFT Training.
# ================================================================================
# Usage:
#     EXP=pickscore bash scripts/reproduce/diffusion_nft_train.sh (on every node)
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP=${EXP:-pickscore}  # geneval, ocr, pickscore, multi_reward
echo "🚀 diffusion_nft training: EXP=${EXP}"

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
prepare_geneval() {
  prepare mask2former
  prepare timm/vit_large_patch14_clip_224.openai
  prepare yifan123/reward-server
  prepare open-mmlab/mmdetection
}
prepare_pickscore() {
  prepare yuvalkirstain/PickScore_v1
  prepare laion/CLIP-ViT-H-14-laion2B-s32B-b79K
}
DATA=${EXP}
REWARD=${EXP}
SERVERS=()
prepare stabilityai/stable-diffusion-3.5-medium
prepare yifan123/flow_grpo
case "$EXP" in
  pickscore)
    prepare_pickscore
    prepare NVlabs/DiffusionNFT  # DrawBench test prompts
    ;;
  ocr)
    prepare paddleocr
    SERVERS=(ocr)
    ;;
  geneval)
    prepare_geneval
    SERVERS=(geneval)
    ;;
  multi_reward)
    # GenEval, PickScore, ClipScore, and HPSv2.1 on the GenEval prompts.
    prepare_geneval
    prepare_pickscore
    prepare openai/clip-vit-large-patch14
    prepare xswu/HPSv2
    DATA=geneval
    SERVERS=(geneval hpsv2)
    ;;
  *) echo "❌ Unknown EXP: ${EXP}"; exit 1 ;;
esac

# --------------------------------------------------------------------------------
# Start the reward servers of this node, and stop them when the training ends.
# --------------------------------------------------------------------------------
stop_servers() {
  for server in "${SERVERS[@]}"; do
    bash scripts/tools/reward_server.sh "$server" stop
  done
}
trap stop_servers EXIT
for server in "${SERVERS[@]}"; do
  bash scripts/tools/reward_server.sh "$server"
done

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
    data=${DATA} \
    algorithm=diffusion_nft \
    model=sd3_5_medium \
    trainer=nft \
    reward=${REWARD} \
    +reproduce=diffusion_nft/${EXP} \
    "$@"
