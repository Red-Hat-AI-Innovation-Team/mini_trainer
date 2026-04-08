#!/usr/bin/env bash
# Multi-node fidelity test for full-state checkpointing.
# Simulates 2 nodes using NUMA-pinned GPU groups: node0=GPUs 0-3, node1=GPUs 4-7.
#
# Usage: ./regression_tests/run_multi_node_fidelity_test.sh [model] [data] [output_dir]

set -euo pipefail

MODEL="${1:-/tmp/tiny_qwen2}"
DATA="${2:-/tmp/tiny_test.jsonl}"
OUTPUT_DIR="${3:-/mnt/nvme3n1/workspace/osilkin/pr-repos/async-checkpointing/mini_trainer/multi_node_fidelity_output}"
TOTAL_STEPS=20
CHECKPOINT_STEP=10
SCRIPT="regression_tests/test_full_state_checkpoint_fidelity.py"
MASTER_PORT=29500

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

run_two_nodes() {
    local mode="$1"
    shift
    echo "=== Running mode=$mode (2 nodes x 4 GPUs) ==="

    # Find a free port to avoid conflicts
    MASTER_PORT=$((29500 + RANDOM % 1000))

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nnodes=2 --nproc-per-node=4 --node-rank=0 \
        --master-addr=localhost --master-port=$MASTER_PORT \
        "$SCRIPT" --model-name-or-path "$MODEL" --data-path "$DATA" \
        --mode "$mode" --total-steps "$TOTAL_STEPS" \
        --output-dir "$OUTPUT_DIR" "$@" &
    PID0=$!

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nnodes=2 --nproc-per-node=4 --node-rank=1 \
        --master-addr=localhost --master-port=$MASTER_PORT \
        "$SCRIPT" --model-name-or-path "$MODEL" --data-path "$DATA" \
        --mode "$mode" --total-steps "$TOTAL_STEPS" \
        --output-dir "$OUTPUT_DIR" "$@" &
    PID1=$!

    # Wait for both nodes
    wait "$PID0"
    STATUS0=$?
    wait "$PID1"
    STATUS1=$?

    if [ $STATUS0 -ne 0 ] || [ $STATUS1 -ne 0 ]; then
        echo "FAIL: mode=$mode failed (node0=$STATUS0, node1=$STATUS1)"
        exit 1
    fi
    echo "=== mode=$mode complete ==="
}

echo "========================================"
echo "Multi-Node Fidelity Test"
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Steps: $TOTAL_STEPS"
echo "  Checkpoint at: step $CHECKPOINT_STEP"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

# Run A: baseline (20 steps straight through)
run_two_nodes baseline

# Run B part 1: checkpoint at step 10
run_two_nodes checkpoint --checkpoint-at-step "$CHECKPOINT_STEP"

# Run B part 2: resume from checkpoint
run_two_nodes resume --checkpoint-at-step "$CHECKPOINT_STEP"

# Compare trajectories
echo "=== Comparing trajectories ==="
python "$SCRIPT" --mode compare --output-dir "$OUTPUT_DIR"
