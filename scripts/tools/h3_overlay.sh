#!/bin/bash
# ================================================================================
# Diffusers Overlay of MiniMax-H3-Image.
# ================================================================================
# Usage, from a launch script that runs an H3-Image experiment:
#   source scripts/tools/h3_overlay.sh
# MiniMax-H3-Image needs a newer diffusers and transformers than this project pins.
# The overlay holds only these two packages, with no dependencies, in
# .venv-h3-overlay. It goes first on PYTHONPATH, so it shadows the two packages of
# .venv for this process only, and the other experiments keep the pinned versions.
# Git ignores the overlay directory. Delete it to install it again.
# ================================================================================

OVERLAY="$(pwd)/.venv-h3-overlay"
DIFFUSERS="git+https://github.com/huggingface/diffusers"
DIFFUSERS_COMMIT=875373ac7f5856071e345a3e394c74efccc6b9e6  # the commit that H3 needs

if [[ -d "$OVERLAY" ]]; then
  echo "✅ H3-Image overlay exists, skipping..."
else
  echo "⏳ Installing the H3-Image overlay..."
  rm -rf "$OVERLAY.tmp"
  uv pip install \
    --python .venv/bin/python --target "$OVERLAY.tmp" --no-deps \
    "transformers==5.22.2" "diffusers @ ${DIFFUSERS}@${DIFFUSERS_COMMIT}"
  mv "$OVERLAY.tmp" "$OVERLAY"
  echo "💾 Prepared the H3-Image overlay"
fi
export PYTHONPATH="${OVERLAY}:${PYTHONPATH}"
