#!/usr/bin/env bash
# TwinTilt local extension: group projection with Euler or SA rollouts.
# EXP=pickscore|geneval|ocr|multi_reward_geneval_clip30
# EXP=pickscore NPROC_PER_NODE=2 bash scripts/reproduce/twintilt_train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESOURCE_ROOT="${TWINTILT_RESOURCE_ROOT:-$(dirname "$REPO_ROOT")}"
PYTHON="$(dirname "$REPO_ROOT")/.venv/bin/python"
EXP="${EXP:-pickscore}"
DATA_EXP="$EXP"
SERVER_NAME=""
case "$EXP" in
  pickscore) ;;
  geneval|ocr) SERVER_NAME="$EXP" ;;
  multi_reward_geneval_clip30) DATA_EXP=geneval; SERVER_NAME=geneval ;;
  *) echo "❌ EXP must be pickscore, geneval, ocr, or multi_reward_geneval_clip30"; exit 1 ;;
esac
[[ -x "$PYTHON" ]] || { echo "❌ Missing project interpreter: $PYTHON"; exit 1; }
cd "$REPO_ROOT"
export TWINTILT_RESOURCE_ROOT="$RESOURCE_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/nexus-align-matplotlib}"

SERVER_PID=""
REWARD_ARGS=()
if [[ "$EXP" == multi_reward_geneval_clip30 ]]; then
  REWARD_ARGS+=("reward.rewards.geneval.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18085}")
elif [[ "$EXP" == geneval ]]; then
  REWARD_ARGS+=("reward.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18085}")
elif [[ "$EXP" == ocr ]]; then
  REWARD_ARGS+=("reward.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18087}")
fi
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT
if [[ -n "$SERVER_NAME" && "${START_REWARD_SERVER:-1}" == 1 ]]; then
  SERVER_SCRIPT=scripts/tools/twintilt_reward_server.sh
  if ! bash "$SERVER_SCRIPT" "$SERVER_NAME" check; then
    mkdir -p logs/reward-servers
    bash "$SERVER_SCRIPT" "$SERVER_NAME" serve >"logs/reward-servers/$SERVER_NAME.log" 2>&1 &
    SERVER_PID=$!
    REWARD_SERVER_PID="$SERVER_PID" bash "$SERVER_SCRIPT" "$SERVER_NAME" wait
  fi
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
echo "🚀 TwinTilt training: $EXP, $NPROC_PER_NODE processes per node"
"$PYTHON" -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  -m nexus_align.cli.main --config-name twintilt \
  "data=twintilt_$DATA_EXP" "reward=twintilt_$EXP" \
  "+reproduce/twintilt=$EXP" "${REWARD_ARGS[@]}" "$@"
