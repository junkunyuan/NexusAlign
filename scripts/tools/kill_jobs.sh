#!/bin/bash
# ================================================================================
# Kill Running Jobs.
# ================================================================================
# Usage:
#   bash scripts/tools/kill_jobs.sh      # list matching processes, confirm, then kill
#   bash scripts/tools/kill_jobs.sh -y   # kill without confirmation
#   bash scripts/tools/kill_jobs.sh -n   # dry run: list only
# Only nexus_align train/eval processes and their launch scripts from this repo are
# targeted (matched by command line and working directory), together with their child
# processes such as torchrun workers, dataloader workers, wandb services, and setup
# steps (uv installs, asset downloads). Other GPU jobs are untouched.
# ================================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PATTERN="nexus_align/cli/main.py|nexus_align\.cli\.main|bin/nexus-align"
PATTERN+="|scripts/reproduce/[a-z0-9_]+_(train|eval)\.sh"  # launchers during setup
GRACE=20  # seconds to wait after SIGTERM before SIGKILL

YES=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    -y) YES=true ;;
    -n) DRY_RUN=true ;;
    *) echo "❌ Unknown option: $arg"; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------------
# Find root processes: match the command line and this repo as the working directory.
# --------------------------------------------------------------------------------
ancestors() {  # PIDs of this script and its parent shells, never to be killed
  local pid=$$
  while [[ "$pid" -gt 1 ]]; do
    echo "$pid"
    pid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
  done
}
SELF=" $(ancestors | tr '\n' ' ') "

ROOTS=()
for pid in $(pgrep -f -- "$PATTERN" || true); do
  [[ "$SELF" == *" $pid "* ]] && continue
  cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$PROJECT_DIR" ]] && ROOTS+=("$pid")
done

if [[ ${#ROOTS[@]} -eq 0 ]]; then
  echo "✅ No running nexus_align jobs found under ${PROJECT_DIR}"
  exit 0
fi

# Collect the full process trees so no worker survives.
descendants() {
  for child in $(pgrep -P "$1" || true); do
    echo "$child"
    descendants "$child"
  done
}
ALL=("${ROOTS[@]}")
for pid in "${ROOTS[@]}"; do
  for child in $(descendants "$pid"); do ALL+=("$child"); done
done
mapfile -t ALL < <(printf "%s\n" "${ALL[@]}" | sort -un)

echo "🖥️ Found ${#ALL[@]} processes (${#ROOTS[@]} roots) to kill:"
ps -o pid=,ppid=,etime=,args= -p "$(IFS=,; echo "${ALL[*]}")" | cut -c1-160

if $DRY_RUN; then
  echo "🛑 Dry run, nothing killed"
  exit 0
fi
if ! $YES; then
  read -r -p "Kill them all? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "🛑 Aborted"; exit 0; }
fi

# --------------------------------------------------------------------------------
# Kill: SIGTERM first so torchrun shuts its workers down, then SIGKILL survivors.
# --------------------------------------------------------------------------------
alive() {
  for pid in "${ALL[@]}"; do kill -0 "$pid" 2>/dev/null && echo "$pid" || true; done
}

echo "🚀 Sending SIGTERM..."
kill -TERM "${ALL[@]}" 2>/dev/null || true
for ((i = 0; i < GRACE; i++)); do
  [[ -z "$(alive)" ]] && break
  echo "⏳ Waiting for processes to exit ($((i + 1))/${GRACE}s)..."
  sleep 1
done

SURVIVORS=($(alive))
if [[ ${#SURVIVORS[@]} -gt 0 ]]; then
  echo "⚠️ ${#SURVIVORS[@]} processes still alive, sending SIGKILL: ${SURVIVORS[*]}"
  kill -KILL "${SURVIVORS[@]}" 2>/dev/null || true
  sleep 2
fi

# --------------------------------------------------------------------------------
# Verify: no matching processes left and GPU memory released.
# --------------------------------------------------------------------------------
if [[ -n "$(alive)" ]]; then
  echo "❌ Failed to kill: $(alive | tr '\n' ' ')"
  exit 1
fi
echo "✅ All nexus_align jobs killed"

if command -v nvidia-smi >/dev/null; then
  APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$APPS" ]]; then
    echo "⚠️ GPU processes still running (not started from this repo):"
    echo "$APPS"
  else
    echo "✅ GPUs are free"
  fi
fi
