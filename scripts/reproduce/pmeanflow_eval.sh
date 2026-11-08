#!/bin/bash
# ================================================================================
# pMeanFlow Evaluation.
# ================================================================================
# Usage:
#   EXP_NAME=<exp_info> CFG_SCALE=8.5 CFG_MIN_T=0.1 CFG_MAX_T=0.7 \
#     bash scripts/reproduce/pmeanflow_eval.sh
# Official CFG settings (eval_config.yml): B/16 8.5 [0.1, 0.7]; L/16 7.0 [0.2, 0.7];
# H/16 7.0 [0.2, 0.6]; B/32 6.5 [0.1, 0.7]; L/32 7.5 [0.2, 0.6]; H/32 5.5 [0.1, 0.6].
# Set RESOLUTION=512 for the /32 models.
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-default}  # name of the run directory in logs/
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
CFG_SCALE=${CFG_SCALE:-8.5}  # guidance scale; 1.0 disables CFG
CFG_MIN_T=${CFG_MIN_T:-0.1}  # guidance interval start
CFG_MAX_T=${CFG_MAX_T:-0.7}  # guidance interval end
NUM_STEPS=${NUM_STEPS:-1}  # sampling steps (NFE)
RESOLUTION=${RESOLUTION:-256}  # image size: 256 or 512
echo "🚀 pmeanflow evaluation: EXP_NAME=${EXP_NAME}" \
  "CFG=${CFG_SCALE} [${CFG_MIN_T}, ${CFG_MAX_T}] NFE=${NUM_STEPS} RESOLUTION=${RESOLUTION}"

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
# Prepare the FID statistics and the Inception weights: download each missing
# asset.
# The 256x256 statistics of pMeanFlow are the same file as those of iMeanFlow.
# --------------------------------------------------------------------------------
case "$RESOLUTION" in
  256) FID_STATS="fid_stats/imf_jax_in256_stats.npz" ;;
  512) FID_STATS="fid_stats/pmf_jax_in512_stats.npz" ;;
  *) echo "❌ Unknown RESOLUTION: ${RESOLUTION}"; exit 1 ;;
esac
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
prepare "$FID_STATS"
prepare torch_fidelity/weights-inception-2015-12-05-6726825d.pth

# --------------------------------------------------------------------------------
# Start the evaluation. The model generates pixels, so the data config has no VAE.
# --------------------------------------------------------------------------------
torchrun \
    --standalone \
    --nproc-per-node ${NPROC} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k \
    data.vae=null \
    job_type=eval \
    evaluator=in1k_50k_${RESOLUTION}_fid_is \
    exp_info=${EXP_NAME} \
    evaluator.cfg_scale=${CFG_SCALE} \
    evaluator.cfg_min_t=${CFG_MIN_T} \
    evaluator.cfg_max_t=${CFG_MAX_T} \
    evaluator.num_steps=${NUM_STEPS} \
    evaluator.balanced_labels=true \
    "$@"
