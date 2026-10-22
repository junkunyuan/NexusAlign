#!/bin/bash
# ================================================================================
# Reward Server.
# ================================================================================
# Usage:
#     bash scripts/tools/reward_server.sh <name>        # start on this node
#     bash scripts/tools/reward_server.sh <name> stop   # stop on this node
# where <name> is ocr, geneval, deqa, hpsv2, imagereward, hpsv3, or unifiedreward.
# A reward model whose dependencies conflict with training runs in its own venv
# (.venv-<name>, built on first use) behind gunicorn on 127.0.0.1, one server per node;
# unifiedreward runs in sglang instead, as Flow-GRPO does.
# Start returns once the server answers, or fails within TIMEOUT seconds.
# ================================================================================

set -e
NAME=$1
ACTION=${2:-start}
TIMEOUT=${TIMEOUT:-900}
NUM_GPUS=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}  # one worker per GPU of this node
case "$NAME" in
  ocr) PORT=18087 ;;
  geneval) PORT=18085 ;;
  deqa) PORT=18086 ;;
  hpsv2) PORT=18090 ;;
  imagereward) PORT=18091 ;;
  hpsv3) PORT=18092 ;;
  unifiedreward) PORT=17140 ;;
  *) echo "❌ Unknown reward server: <$NAME>; use ocr, geneval, deqa, hpsv2," \
       "imagereward, hpsv3, or unifiedreward"; exit 1 ;;
esac

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${PROJECT_DIR}/data_and_model"
VENV="${PROJECT_DIR}/.venv-${NAME}"
SERVERS="${PROJECT_DIR}/src/nexus_align/rewards/servers"
LOG_DIR="${PROJECT_DIR}/logs/reward_server"
LOG="${LOG_DIR}/${NAME}-$(hostname).log"
PID_FILE="${LOG_DIR}/${NAME}-$(hostname).pid"
URL="http://127.0.0.1:${PORT}"
mkdir -p "$LOG_DIR"

is_up() {  # a GET on the POST-only route returns 405 once a worker serves requests
  if [[ "$NAME" == "unifiedreward" ]]; then  # sglang logs this once its warmup passes
    [[ "$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 5 "$URL/health")" == "200" ]] \
      && grep -q "ready to roll" "$LOG"
  else
    [[ "$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 5 "$URL")" == "405" ]]
  fi
}

# --------------------------------------------------------------------------------
# Stop.
# --------------------------------------------------------------------------------
if [[ "$ACTION" == "stop" ]]; then
  # sglang leads its own process group, so its workers stop with it; it can hang while
  # draining requests, so the group is killed if it outlives the grace period.
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -- "-$PID" 2>/dev/null || kill "$PID" 2>/dev/null; then
      for (( i = 0; i < 30 && $(pgrep -g "$PID" -c || true) > 0; i++ )); do sleep 1; done
      kill -9 -- "-$PID" 2>/dev/null || true
      echo "🛑 Stopped ${NAME} reward server"
    fi
  fi
  rm -f "$PID_FILE"
  exit 0
fi

if is_up; then
  echo "✅ ${NAME} reward server is already up at ${URL}"
  exit 0
fi

# --------------------------------------------------------------------------------
# Prepare environment.
# --------------------------------------------------------------------------------
if [[ -f "${VENV}/.installed" ]]; then
  echo "Reward environment installed, skipping..."
else
  echo "Installing ${NAME} reward environment..."
  rm -rf "$VENV"  # drop a partial install
  command -v uv >/dev/null || pip install uv
  UV="$(command -v uv || echo "${HOME}/.local/bin/uv")"
  # torch 2.1.2 has no wheels for Python releases newer than 3.11.
  "$UV" venv --python 3.11 "$VENV"
  UV_PIP="${UV} pip install --python ${VENV}/bin/python"
  # torch and its CUDA libraries come from the PyTorch index.
  PT_PIP="${UV} pip install --python ${VENV}/bin/python -i"
  PT=https://download.pytorch.org/whl
  if [[ "$NAME" == "ocr" ]]; then
    # Pin the deps whose newest releases need numpy 2, so uv does not backtrack through
    # their large wheels.
    $UV_PIP paddlepaddle==2.6.2 paddleocr==2.9.1 python-Levenshtein flask gunicorn "setuptools<70" \
      numpy==1.26.4 opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86 \
      opencv-python-headless==4.11.0.86 scipy==1.17.1 tifffile==2026.3.3 contourpy==1.3.3
    # paddleocr pulls in GUI builds of OpenCV that need libGL; keep one headless build.
    "$UV" pip uninstall --python "${VENV}/bin/python" opencv-python opencv-contrib-python \
      opencv-python-headless
    $UV_PIP opencv-contrib-python-headless==4.11.0.86 numpy==1.26.4
  elif [[ "$NAME" == "hpsv2" || "$NAME" == "imagereward" ]]; then
    # Versions follow DiffusionNFT's setup.py and README.
    $PT_PIP ${PT}/cu126 torch==2.6.0+cu126 torchvision==0.21.0+cu126
    NFT_DEPS="transformers==4.40.0 tokenizers==0.19.1 huggingface-hub==0.29.1 \
      diffusers==0.33.1 accelerate==1.4.0 datasets==3.3.2 pyarrow==19.0.1 numpy==1.26.4 \
      flask==3.1.0 gunicorn==23.0.0"
    if [[ "$NAME" == "hpsv2" ]]; then
      $UV_PIP hpsv2x==1.2.0 $NFT_DEPS
    else
      $UV_PIP image-reward==1.5 $NFT_DEPS \
        "clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
    fi
  elif [[ "$NAME" == "deqa" ]]; then
    # torch follows reward-server, and the rest DeQA-Score's pyproject.toml.
    $PT_PIP ${PT}/cu121 torch==2.1.2 torchvision==0.16.2
    $UV_PIP torch==2.1.2 torchvision==0.16.2 transformers==4.36.1 tokenizers==0.15.0 \
      sentencepiece==0.1.99 accelerate==0.21.0 icecream einops==0.6.1 numpy==1.26.0 \
      protobuf flask==3.1.0 gunicorn==23.0.0
  elif [[ "$NAME" == "hpsv3" ]]; then
    # Versions follow HPSv3's requirements.txt; deepspeed is only needed for training.
    $PT_PIP ${PT}/cu126 torch==2.6.0+cu126 torchvision==0.21.0+cu126
    $UV_PIP --no-deps hpsv3==1.0.0
    $UV_PIP transformers==4.45.2 tokenizers==0.20.3 accelerate==1.8.0 peft==0.10.0 trl==0.8.6 \
      rich==14.0.0 datasets==3.6.0 fire==0.7.0 omegaconf==2.3.0 qwen-vl-utils==0.0.11 \
      einops==0.8.1 safetensors==0.5.3 matplotlib==3.10.3 pandas==2.3.0 tensorboard==2.19.0 \
      opencv-python-headless==4.11.0.86 numpy==2.1.2 flask==3.1.0 gunicorn==23.0.0
  elif [[ "$NAME" == "unifiedreward" ]]; then
    # Flow-GRPO installs the sglang of its time unpinned; compressed-tensors is pinned to
    # a release that still works with the transformers that sglang pins.
    $PT_PIP ${PT}/cu124 torch==2.6.0 torchvision==0.21.0  # the builds sglang pins
    $UV_PIP "sglang[all]==0.4.6.post5" compressed-tensors==0.9.4
  else
    # Versions follow reward-server; opencv is headless as no display libs are needed.
    $PT_PIP ${PT}/cu121 torch==2.1.2 torchvision==0.16.2
    $UV_PIP torch==2.1.2 torchvision==0.16.2 numpy==1.26.0 gunicorn==23.0.0 \
      open-clip-torch==2.31.0 opencv-python-headless==4.11.0.86 clip-benchmark==1.6.1 \
      flask==3.1.0 pandas==2.2.3 "setuptools<70" wheel ninja
    ARCH=$("${VENV}/bin/python" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")
    TORCH_CUDA_ARCH_LIST=$ARCH MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=32 \
      $UV_PIP --no-build-isolation mmcv-full==1.7.2
    $UV_PIP --no-build-isolation "${DATA}/open-mmlab/mmdetection"
    # mmcv-full pulls in opencv-python, which needs libGL; keep only the headless build.
    "$UV" pip uninstall --python "${VENV}/bin/python" opencv-python
    $UV_PIP --reinstall-package opencv-python-headless opencv-python-headless==4.11.0.86 numpy==1.26.0
  fi
  touch "${VENV}/.installed"
  echo "Installed ${NAME} reward environment"
fi

# --------------------------------------------------------------------------------
# Start the server and wait until it answers.
# --------------------------------------------------------------------------------
if [[ "$NAME" == "ocr" ]]; then
  APP_DIR="$SERVERS"
  APP="ocr_server:create_app('${DATA}/paddleocr/whl')"
elif [[ "$NAME" == "deqa" ]]; then
  APP_DIR="$SERVERS"
  APP="deqa_server:create_app('${DATA}/zhiyuanyou/DeQA-Score-Mix3')"
elif [[ "$NAME" == "hpsv2" ]]; then
  APP_DIR="$SERVERS"
  APP="hpsv2_server:create_app('${DATA}/xswu/HPSv2/HPS_v2.1_compressed.pt')"
elif [[ "$NAME" == "imagereward" ]]; then
  APP_DIR="$SERVERS"
  APP="imagereward_server:create_app('${DATA}/THUDM/ImageReward', \
'${DATA}/google-bert/bert-base-uncased')"
elif [[ "$NAME" == "hpsv3" ]]; then
  APP_DIR="$SERVERS"
  APP="hpsv3_server:create_app('${DATA}/MizzenAI/HPSv3/HPSv3.safetensors', \
'${DATA}/Qwen/Qwen2-VL-7B-Instruct')"
else
  APP_DIR="${DATA}/yifan123/reward-server"
  CONFIG="${DATA}/open-mmlab/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
  CLIP="${DATA}/timm/vit_large_patch14_clip_224.openai/open_clip_model.safetensors"
  APP="geneval_server:create_app('${APP_DIR}', '${CONFIG}', '${DATA}/mask2former', '${CLIP}')"
fi
echo "🚀 Starting ${NAME} reward server at ${URL}, log: <${LOG}>"
if [[ "$NAME" == "unifiedreward" ]]; then
  # The official launch of Flow-GRPO, data parallel over this node's GPUs; the smaller
  # static memory leaves room for sampling on the same GPUs. It runs in data_and_model/
  # so that the vision tower named in the model config resolves to the local copy, and
  # its warmup requests to itself skip the proxy.
  setsid nohup env -C "$DATA" no_proxy="127.0.0.1,localhost,${no_proxy:-}" \
    "${VENV}/bin/python" -m sglang.launch_server \
    --model-path "${DATA}/CodeGoat24/UnifiedReward-7b-v1.5" \
    --served-model-name UnifiedReward-7b-v1.5 --api-key flowgrpo --host 127.0.0.1 \
    --port $PORT --chat-template chatml-llava --enable-p2p-check \
    --mem-fraction-static 0.3 --dp-size ${NUM_GPUS} > "$LOG" 2>&1 < /dev/null &
else
  EXTRA_ARGS=()
  if [[ "$NAME" == "deqa" || "$NAME" == "hpsv3" ]]; then
    EXTRA_ARGS+=(--timeout 900)  # a worker loads its 7B model before it answers
  fi
  REWARD_SERVER_PORT=$PORT REWARD_SERVER_WORKERS=${NUM_GPUS} nohup \
    "${VENV}/bin/gunicorn" -c "${SERVERS}/gunicorn.conf.py" "${EXTRA_ARGS[@]}" \
    --chdir "$APP_DIR" --pythonpath "$SERVERS" "$APP" > "$LOG" 2>&1 < /dev/null &
fi
echo $! > "$PID_FILE"

for (( i = 0; i < TIMEOUT; i += 5 )); do
  if is_up; then
    echo "✅ ${NAME} reward server is up at ${URL}"
    exit 0
  fi
  # sglang's launcher outlives its failed workers, so its log is checked too.
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null \
      || grep -q "Child process unexpectedly failed" "$LOG"; then
    echo "❌ ${NAME} reward server exited; see <${LOG}>"
    tail -20 "$LOG"
    bash "$0" "$NAME" stop
    exit 1
  fi
  sleep 5
done
echo "❌ ${NAME} reward server not up after ${TIMEOUT}s; see <${LOG}>"
bash "$0" "$NAME" stop
exit 1
