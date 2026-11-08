#!/bin/bash
# ================================================================================
# PixelDiT2 Evaluation.
# ================================================================================
# Usage:
#   EXP_NAME=<exp_info> bash scripts/reproduce/pixeldit2_eval.sh
# Official c2i/README.md: 50 Heun steps; H/16 CFG 2.4 on [0.125, 0.9] at 256x256, and
# CFG 2.1 on [0.1, 0.925] at 512x512 (RESOLUTION=512). t = 1 is the clean image.
# ================================================================================

set -e
# --------------------------------------------------------------------------------
# Prepare the experiment.
# --------------------------------------------------------------------------------
EXP_NAME=${EXP_NAME:-default}  # name of the run directory in logs/
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
CFG_SCALE=${CFG_SCALE:-2.4}  # guidance scale; 1.0 disables CFG
CFG_MIN_T=${CFG_MIN_T:-0.125}  # guidance interval start
CFG_MAX_T=${CFG_MAX_T:-0.9}  # guidance interval end
NUM_STEPS=${NUM_STEPS:-50}  # ODE steps: Heun, then one Euler step at the end
RESOLUTION=${RESOLUTION:-256}  # image size: 256 or 512
echo "🚀 pixeldit2 evaluation: EXP_NAME=${EXP_NAME} CFG=${CFG_SCALE}" \
  "[${CFG_MIN_T}, ${CFG_MAX_T}] STEPS=${NUM_STEPS} RESOLUTION=${RESOLUTION}"

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
# Prepare the FID statistics, the Inception weights, and the grounding encoder:
# download each missing asset.
# --------------------------------------------------------------------------------
case "$RESOLUTION" in
  256)
    FID_STATS="fid_stats/jit_in256_stats.npz"
    ENCODER="timm/vit_small_patch16_dinov3_qkvb.lvd1689m"
    ;;
  512)
    FID_STATS="fid_stats/jit_in512_stats.npz"
    ENCODER="timm/vit_base_patch16_dinov3_qkvb.lvd1689m"
    ;;
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
prepare "$ENCODER"

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
    evaluator.fid_stats="\${data_and_model_dir}/${FID_STATS}" \
    exp_info=${EXP_NAME} \
    evaluator.cfg_scale=${CFG_SCALE} \
    evaluator.cfg_min_t=${CFG_MIN_T} \
    evaluator.cfg_max_t=${CFG_MAX_T} \
    evaluator.num_steps=${NUM_STEPS} \
    evaluator.balanced_labels=true \
    "$@"
