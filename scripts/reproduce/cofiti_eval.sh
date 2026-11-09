#!/usr/bin/env bash
# Evaluate a saved CoFiTi experiment with one task's test dataset and rewards.
# EXP=pickscore|geneval|ocr|multi_reward_geneval_clip30
# EXP_NAME=<training directory name under logs/> NPROC_PER_NODE=1 bash .../cofiti_eval.sh
# Hydra overrides are forwarded unchanged, for example:
#   evaluator.ckpts=[/absolute/path/to/checkpoint] evaluator.new_run=true
#   evaluator.num_prompts=2 evaluator.num_repeats=1
#   evaluator.num_steps=3 evaluator.resolution=256
# Defaults: student, 40 Flow Euler steps, 512 px, CFG 1, T5 256, no churn.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESOURCE_ROOT="${COFITI_RESOURCE_ROOT:-$(dirname "$REPO_ROOT")}"
PYTHON="$(dirname "$REPO_ROOT")/.venv/bin/python"
EXP="${EXP:-pickscore}"
DATA_EXP="$EXP"
SERVER_NAME=""
: "${EXP_NAME:?Set EXP_NAME to the training directory name under logs/}"
case "$EXP" in
  pickscore) ;;
  geneval|ocr) SERVER_NAME="$EXP" ;;
  multi_reward_geneval_clip30) DATA_EXP=geneval; SERVER_NAME=geneval ;;
  *) echo "❌ EXP must be pickscore, geneval, ocr, or multi_reward_geneval_clip30"; exit 1 ;;
esac
[[ -x "$PYTHON" ]] || {
  echo "❌ Missing project interpreter: $PYTHON" >&2
  exit 1
}
cd "$REPO_ROOT"
export COFITI_RESOURCE_ROOT="$RESOURCE_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/nexus-align-matplotlib}"

SERVER_PID=""
REWARD_ARGS=()
if [[ "$EXP" == multi_reward_geneval_clip30 ]]; then
  REWARD_ARGS+=("rewards.$EXP.rewards.geneval.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18085}")
elif [[ "$EXP" == geneval ]]; then
  REWARD_ARGS+=("rewards.$EXP.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18085}")
elif [[ "$EXP" == ocr ]]; then
  REWARD_ARGS+=("rewards.$EXP.url=http://127.0.0.1:${REWARD_SERVER_PORT:-18087}")
fi
CLI_ARGS=(
  --config-name cofiti
  job_type=eval "exp_info=$EXP_NAME" "data=cofiti_$DATA_EXP"
  "+reward@rewards.$EXP=cofiti_$EXP" "${REWARD_ARGS[@]}" "$@"
)

# Hydra inspection commands need neither a saved experiment nor reward services.
for arg in "$@"; do
  case "$arg" in
    --help|-h|--hydra-help|--version|--cfg|-c|--cfg=*|--info|-i|--info=*)
      exec "$PYTHON" -m nexus_align.cli.main "${CLI_ARGS[@]}"
      ;;
  esac
done

# Use Hydra itself so log_dir/exp_info overrides and interpolations match the job.
"$PYTHON" - "$REPO_ROOT" "${CLI_ARGS[@]}" <<'PY'
import os
from pathlib import Path
import sys

import hydra

repo_root = Path(sys.argv.pop(1))


@hydra.main(
    config_path=str(repo_root / "src/nexus_align/configs"),
    config_name="cofiti",
    version_base="1.3",
)
def validate_experiment(cfg):
    if cfg.job_type != "eval":
        raise SystemExit("❌ cofiti_eval.sh requires job_type=eval")
    experiment = Path(os.path.abspath(os.path.join(cfg.log_dir, cfg.exp_info)))
    if not experiment.is_dir():
        raise SystemExit(f"❌ Evaluation experiment directory not found: {experiment}")
    config = experiment / "config_train.yaml"
    if not config.is_file():
        raise SystemExit(f"❌ Evaluation requires the saved training config: {config}")
    print(f"✅ Evaluation experiment: {experiment}")


validate_experiment()
PY

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT
if [[ -n "$SERVER_NAME" && "${START_REWARD_SERVER:-1}" == 1 ]]; then
  SERVER_SCRIPT=scripts/tools/cofiti_reward_server.sh
  if ! bash "$SERVER_SCRIPT" "$SERVER_NAME" check; then
    mkdir -p logs/reward-servers
    bash "$SERVER_SCRIPT" "$SERVER_NAME" serve >"logs/reward-servers/$SERVER_NAME.log" 2>&1 &
    SERVER_PID=$!
    REWARD_SERVER_PID="$SERVER_PID" bash "$SERVER_SCRIPT" "$SERVER_NAME" wait
  fi
fi
"$PYTHON" -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29501}" \
  -m nexus_align.cli.main "${CLI_ARGS[@]}"
