#!/bin/bash
# ================================================================================
# TwinTilt Evaluation.
# ================================================================================
# The protocol follows the evaluation of Flow-GRPO: the EMA weights, the Euler ODE with
# 40 steps, CFG 4.5, and 128 T5 tokens.
# Usage:
#   EXP_NAME=<exp_info> DATA=geneval REWARDS="geneval twintilt_multi" \
#     bash scripts/reproduce/twintilt_eval.sh
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-default}  # name of the run directory in logs/
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
echo "🚀 twintilt evaluation: EXP_NAME=${EXP_NAME}"

# --------------------------------------------------------------------------------
# Prepare the environment.
# --------------------------------------------------------------------------------
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export NCCL_DEBUG=VERSION  # print only the NCCL version

echo "🖥️ NPROC_PER_NODE=${NPROC}"

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

for arg in "$@"; do
  if [[ "$arg" =~ ^(EXP_NAME|DATA|REWARDS|NPROC)= ]]; then
    echo "❌ Set ${arg%%=*} before bash, not after the script: ${arg%%=*}=... bash $0"
    exit 1
  fi
done
CONFIG=logs/${EXP_NAME}/config_train.yaml
if [[ ! -f "$CONFIG" ]]; then
  echo "❌ Training config not found: ${CONFIG}; set EXP_NAME to a run under logs/"
  exit 1
fi
REWARDS=${REWARDS//,/ }  # commas also work
if [[ -z "$DATA" || -z "${REWARDS// /}" ]]; then
  echo "❌ Set DATA and REWARDS, e.g. DATA=pickscore REWARDS=\"pickscore" \
    "twintilt_pickscore\", or DATA=geneval REWARDS=\"geneval twintilt_multi\""
  exit 1
fi
if [[ " geneval ocr pickscore drawbench " != *" ${DATA} "* ]]; then
  echo "❌ DATA=${DATA} is not an image test set; use geneval, ocr, pickscore," \
    "or drawbench"
  exit 1
fi
for reward in $REWARDS; do
  target=$reward
  [[ "$reward" == twintilt_multi ]] && target=geneval
  if [[ " ocr geneval " == *" ${target} "* && "$target" != "$DATA" ]]; then
    echo "❌ The ${reward} reward needs DATA=${target}: it checks each prompt's target"
    exit 1
  fi
  if [[ " ${REWARDS} " == *" twintilt_multi "* \
      && " twintilt_pickscore hpsv2 clipscore geneval " == *" ${reward} "* ]]; then
    echo "❌ twintilt_multi already reports ${reward}; evaluate them in separate runs"
    exit 1
  fi
done
echo "🚀 twintilt evaluation: DATA=${DATA} REWARDS=${REWARDS}"
EVAL_ARGS=()
if [[ " ${REWARDS} " == *" geneval "* ]]; then
  EVAL_ARGS+=(rewards.geneval.jpeg=false)  # TwinTilt scores lossless PNG images
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
prepare stabilityai/stable-diffusion-3.5-medium
prepare yifan123/flow_grpo
prepare NVlabs/DiffusionNFT  # DrawBench test prompts
SERVERS=()
for reward in $REWARDS; do
  case "$reward" in
    ocr) prepare paddleocr ;;
    geneval) prepare_geneval ;;
    pickscore | twintilt_pickscore) prepare_pickscore ;;
    qwen_judge) prepare Qwen/Qwen3.8-27B ;;
    twintilt_multi)
      prepare_pickscore
      prepare xswu/HPSv2
      prepare openai/clip-vit-large-patch14
      prepare_geneval
      SERVERS+=(geneval hpsv2)
      ;;
    aesthetic)
      prepare openai/clip-vit-large-patch14
      prepare aesthetic
      ;;
    clipscore) prepare openai/clip-vit-large-patch14 ;;
    hpsv2) prepare xswu/HPSv2 ;;
    imagereward)
      prepare THUDM/ImageReward
      prepare google-bert/bert-base-uncased
      ;;
    deqa) prepare zhiyuanyou/DeQA-Score-Mix3 ;;
    unifiedreward)
      prepare CodeGoat24/UnifiedReward-7b-v1.5
      prepare google/siglip-so400m-patch14-384
      ;;
    hpsv3)
      prepare MizzenAI/HPSv3
      prepare Qwen/Qwen2-VL-7B-Instruct
      ;;
    *) echo "❌ Unknown reward: ${reward}"; exit 1 ;;
  esac
  if [[ " ocr geneval deqa hpsv2 imagereward hpsv3 qwen_judge unifiedreward " \
      == *" ${reward} "* ]]; then
    SERVERS+=("$reward")
  fi
  EVAL_ARGS+=("+reward@rewards.${reward}=${reward}")
done

# --------------------------------------------------------------------------------
# Start the reward servers of this node, and stop them when the evaluation ends.
# --------------------------------------------------------------------------------
SERVERS=($(printf "%s\n" "${SERVERS[@]}" | awk '!seen[$0]++'))  # drop duplicates
# sglang sizes its memory pool from the free GPU memory, so it starts first.
if [[ " ${SERVERS[*]} " == *" unifiedreward "* ]]; then
  OTHERS=()
  for server in "${SERVERS[@]}"; do
    [[ "$server" == unifiedreward ]] || OTHERS+=("$server")
  done
  SERVERS=(unifiedreward "${OTHERS[@]}")
fi
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
# Start the evaluation.
# --------------------------------------------------------------------------------
torchrun \
    --standalone \
    --nproc-per-node ${NPROC} \
    src/nexus_align/cli/main.py \
    data=${DATA} \
    job_type=eval \
    evaluator=reward \
    exp_info=${EXP_NAME} \
    "${EVAL_ARGS[@]}" \
    "$@"
