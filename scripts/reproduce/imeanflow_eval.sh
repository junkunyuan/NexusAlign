#!/bin/bash
# ================================================================================
# iMeanFlow Evaluation.
# ================================================================================
# Usage:
#   EXP_NAME=<exp_info> CFG_SCALE=8.0 CFG_MIN_T=0.4 CFG_MAX_T=0.65 \
#     bash scripts/reproduce/imeanflow_eval.sh
# Official CFG settings: B/2 8.0 [0.4, 0.65]; M/2, L/2 10.5 [0.4, 0.6];
# XL/2 8.0 [0.42, 0.62]; XL/2 2-NFE 4.0 [0.36, 0.64].
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-default}  # name of the run directory in logs/
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
CFG_SCALE=${CFG_SCALE:-8.0}  # guidance scale; 1.0 disables CFG
CFG_MIN_T=${CFG_MIN_T:-0.4}  # guidance interval start
CFG_MAX_T=${CFG_MAX_T:-0.65}  # guidance interval end
NUM_STEPS=${NUM_STEPS:-1}  # sampling steps (NFE)
echo "🚀 imeanflow evaluation: EXP_NAME=${EXP_NAME} CFG=${CFG_SCALE} [${CFG_MIN_T}, ${CFG_MAX_T}] NFE=${NUM_STEPS}"

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
    evaluator.cfg_scale=${CFG_SCALE} \
    evaluator.cfg_min_t=${CFG_MIN_T} \
    evaluator.cfg_max_t=${CFG_MAX_T} \
    evaluator.num_steps=${NUM_STEPS} \
    evaluator.balanced_labels=true \
    "$@"
