#!/usr/bin/env bash
# Run the existing local GenEval/OCR scorer in the parent project's environment.
# Usage: cofiti_reward_server.sh geneval|ocr [serve|check|wait]
# Serve stays in the foreground; the training launcher owns and cleans up its PID.
set -euo pipefail

NAME=${1:?Usage: cofiti_reward_server.sh geneval|ocr [serve|check|wait]}
ACTION=${2:-serve}
case "$NAME" in
  geneval) DEFAULT_PORT=18085 ;;
  ocr) DEFAULT_PORT=18087 ;;
  *) echo "❌ Unknown CoFiTi reward server: $NAME" >&2; exit 2 ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export COFITI_RESOURCE_ROOT="${COFITI_RESOURCE_ROOT:-$(dirname "$PROJECT_DIR")}"
PYTHON="$(dirname "$PROJECT_DIR")/.venv/bin/python"
PORT=${REWARD_SERVER_PORT:-$DEFAULT_PORT}
TIMEOUT=${REWARD_SERVER_TIMEOUT:-900}
URL="http://127.0.0.1:${PORT}"

is_up() {
  local status
  status=$(curl --silent --noproxy '*' --output /dev/null --write-out '%{http_code}' \
    --max-time 5 "$URL/" || true)
  [[ "$status" == "405" ]]
}

case "$ACTION" in
  check) is_up; exit $? ;;
  wait)
    deadline=$((SECONDS + TIMEOUT))
    until is_up; do
      if ((SECONDS >= deadline)); then
        echo "❌ ${NAME} reward server did not become ready at ${URL}" >&2
        exit 1
      fi
      if [[ -n "${REWARD_SERVER_PID:-}" ]] \
        && ! kill -0 "$REWARD_SERVER_PID" 2>/dev/null; then
        echo "❌ ${NAME} reward server exited before becoming ready" >&2
        exit 1
      fi
      sleep 1
    done
    echo "✅ ${NAME} reward server is ready at ${URL}"
    exit 0
    ;;
  serve) ;;
  *) echo "❌ Unknown action: $ACTION; use serve, check, or wait" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "❌ Project Python not found: ${PYTHON}" >&2
  exit 1
fi
if is_up; then
  echo "✅ Reusing ${NAME} reward server at ${URL}"
  exit 0
fi

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Do not use the shared gunicorn config: it replaces CUDA_VISIBLE_DEVICES.
echo "🚀 Starting ${NAME} reward server at ${URL}"
exec "$PYTHON" -m gunicorn \
  --bind "127.0.0.1:${PORT}" \
  --workers "${REWARD_SERVER_WORKERS:-1}" \
  --timeout "$TIMEOUT" \
  --graceful-timeout 30 \
  "nexus_align.rewards.servers.${NAME}_server:create_local_app()"
