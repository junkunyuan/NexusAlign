#!/bin/bash
# Quick smoke tests for Qwen-Image, Z-Image, and SD3-medium pipelines.
# Each test runs ~10 training steps at 64x64 with batch_size=2 and group_size=2,
# validates every 5 steps using normalized edit distance reward on text_rendering_benchmark.
#
# Prerequisites:
#   cd /HOME/neuq_mxcui/neuq_mxcui_1/HDD_POOL/NexusAlign
#   pip install -e .
#
# Usage:
#   bash tests/run_quick_tests.sh              # run all 3 tests on 1 GPU
#   bash tests/run_quick_tests.sh 2            # run all 3 tests on 2 GPUs
#   bash tests/run_quick_tests.sh 1 sd3        # run only SD3 test on 1 GPU

set -e

NGPU=${1:-4}
TEST_FILTER=${2:-all}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "  NexusAlign Quick Test Suite"
echo "  GPUs: $NGPU  |  Filter: $TEST_FILTER"
echo "========================================"

run_test() {
    local model_name=$1
    local model_default=$2
    local config_file=$3

    echo ""
    echo "========================================"
    echo "  Testing: $model_name"
    echo "========================================"
    echo ""

    nexus-align train \
        --nproc_per_node="$NGPU" \
        -c "$config_file" \
        model="$model_default" \
        data=text_rendering \
        reward_model=normalized_edit_distance

    echo ""
    echo "✅ $model_name test PASSED"
    echo ""
}

# ---- Qwen-Image ----
if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "qwen_image" ]]; then
    run_test "Qwen-Image" "qwen_image" "tests/quick_test_qwen_image.yaml"
fi

# ---- Z-Image ----
if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "z_image" ]]; then
    run_test "Z-Image" "z_image" "tests/quick_test_z_image.yaml"
fi

# ---- SD3-medium ----
if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "sd3" ]]; then
    run_test "SD3-medium" "sd3" "tests/quick_test_sd3.yaml"
fi

echo ""
echo "========================================"
echo "  All selected tests PASSED"
echo "========================================"
