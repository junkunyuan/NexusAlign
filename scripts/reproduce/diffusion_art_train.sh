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
EXP=${EXP:-pickscore}  # geneval, ocr, pickscore, multi_reward, qwen_judge, h3_hpsv3
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
MODEL=sd3_5_medium
ALGORITHM=diffusion_art
TRAINER=art
SERVERS=()
case "$EXP" in
  pickscore)
    prepare_pickscore
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
  qwen_judge)
    # One judge per node, across its GPUs: 27.8 B parameters do not fit per rank.
    prepare Qwen/Qwen3.8-27B
    DATA=pickscore
    SERVERS=(qwen_judge)
    ;;
  h3_hpsv3)
    prepare MiniMaxAI/MiniMax-H3-Image
    prepare MizzenAI/HPSv3
    prepare Qwen/Qwen2-VL-7B-Instruct
    DATA=pickscore
    REWARD=hpsv3
    MODEL=minimax_h3_image
    ALGORITHM=${ALGORITHM}_minimax_h3_image
    TRAINER=diffusion_art_minimax_h3_image
    SERVERS=(hpsv3)
    source scripts/tools/h3_overlay.sh
    ;;
  *) echo "❌ Unknown EXP: ${EXP}"; exit 1 ;;
esac
if [[ "$MODEL" == "sd3_5_medium" ]]; then
  prepare stabilityai/stable-diffusion-3.5-medium
  prepare yifan123/flow_grpo
fi

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
    algorithm=${ALGORITHM} \
    model=${MODEL} \
    trainer=${TRAINER} \
    reward=${REWARD} \
    +reproduce=diffusion_art/${EXP} \
    "$@"
