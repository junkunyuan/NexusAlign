#!/bin/bash
# ================================================================================
# MeanFlow Evaluation.
# ================================================================================
# Usage:
#   EXP_NAME=<exp_info> bash scripts/reproduce/meanflow_eval.sh
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-default}  # name of the run directory in logs/
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
echo "🚀 meanflow evaluation: EXP_NAME=${EXP_NAME}"

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

# --------------------------------------------------------------------------------
# Prepare the FID statistics, the Inception weights, and the VAE: download each
# missing asset.
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
prepare fid_stats/imf_jax_in256_stats.npz
prepare torch_fidelity/weights-inception-2015-12-05-6726825d.pth
prepare stabilityai/sd-vae-ft-mse

# --------------------------------------------------------------------------------
# Start the evaluation.
# --------------------------------------------------------------------------------
torchrun \
    --standalone \
    --nproc-per-node ${NPROC} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k \
    job_type=eval \
    evaluator=in1k_50k_256_fid_is \
    exp_info=${EXP_NAME} \
    "$@"
