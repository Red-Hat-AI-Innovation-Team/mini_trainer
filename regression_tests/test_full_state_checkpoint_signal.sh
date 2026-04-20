#!/usr/bin/env bash
# Test that writing the trigger file causes training to checkpoint and exit cleanly.
#
# Usage: ./regression_tests/test_full_state_checkpoint_signal.sh [model] [data]

set -euo pipefail

MODEL="${1:-/tmp/tiny_qwen2}"
DATA="${2:-/tmp/tiny_test.jsonl}"
OUTPUT_DIR="${3:-/tmp/signal_test_output}"
TRIGGER_FILE="/dev/shm/mini_trainer_checkpoint_trigger"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
rm -f "$TRIGGER_FILE"

echo "========================================"
echo "Signal-Triggered Checkpoint Test"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

echo "=== Phase 1: Start training with on-demand checkpointing ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
    -m mini_trainer.train \
    --model-name-or-path "$MODEL" \
    --data-path "$DATA" \
    --batch-size 2 \
    --max-tokens-per-gpu 512 \
    --learning-rate 1e-4 \
    --output-dir "$OUTPUT_DIR" \
    --training-mode infinite \
    --osft \
    --osft-unfreeze-rank-ratio 0.2 \
    --on-demand-checkpointing \
    > "$OUTPUT_DIR/train.log" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Wait for training to produce at least 5 metric lines
echo "Waiting for training to start..."
STARTED=false
for i in $(seq 1 180); do
    METRICS_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'training_metrics_*.jsonl' -print -quit 2>/dev/null)
    if [ -n "$METRICS_FILE" ]; then
        LINES=$(wc -l < "$METRICS_FILE")
        if [ "$LINES" -ge 5 ]; then
            echo "Training running — $LINES steps logged"
            STARTED=true
            break
        fi
    fi
    # Check if process died
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "FAIL: Training process died before producing steps"
        cat "$OUTPUT_DIR/train.log" | tail -20
        exit 1
    fi
    sleep 2
done

if [ "$STARTED" = false ]; then
    echo "FAIL: Training did not start within timeout"
    kill -9 "$TRAIN_PID" 2>/dev/null
    exit 1
fi

# Write trigger file (this is what signal handlers do in production)
echo "=== Writing trigger file: $TRIGGER_FILE ==="
touch "$TRIGGER_FILE"

# Wait for clean exit
echo "Waiting for training to checkpoint and exit..."
EXITED=false
for i in $(seq 1 120); do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        EXIT_CODE=0
        wait "$TRAIN_PID" 2>/dev/null || EXIT_CODE=$?
        echo "Training exited with code: $EXIT_CODE"
        EXITED=true
        break
    fi
    sleep 2
done

if [ "$EXITED" = false ]; then
    echo "FAIL: Training did not exit within timeout"
    kill -9 "$TRAIN_PID" 2>/dev/null
    exit 1
fi

echo ""
echo "=== Phase 2: Verify checkpoint structure ==="

CKPT_DIR="$OUTPUT_DIR/full_state_checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "FAIL: No checkpoint directory at $CKPT_DIR"
    exit 1
fi

STEP_DIR=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | head -1)
if [ -z "$STEP_DIR" ]; then
    echo "FAIL: No step directory in $CKPT_DIR"
    exit 1
fi

echo "Checkpoint directory: $STEP_DIR"

# Check DCP distributed directory
if [ ! -d "$STEP_DIR/distributed" ]; then
    echo "FAIL: No distributed/ directory in checkpoint"
    exit 1
fi
echo "  OK: distributed/ directory exists"

# Check metadata
if [ ! -f "$STEP_DIR/training_state.pt" ]; then
    echo "FAIL: No training_state.pt"
    exit 1
fi
echo "  OK: training_state.pt exists"

# Check per-rank RNG files
for r in 0 1 2 3; do
    if [ ! -f "$STEP_DIR/rng_state_rank_${r}.pt" ]; then
        echo "FAIL: Missing rng_state_rank_${r}.pt"
        exit 1
    fi
done
echo "  OK: rng_state_rank_{0..3}.pt all present"

# Check DCP shard files
SHARD_COUNT=$(ls "$STEP_DIR/distributed/"__*_*.distcp 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -lt 4 ]; then
    echo "FAIL: Expected at least 4 DCP shard files, got $SHARD_COUNT"
    exit 1
fi
echo "  OK: $SHARD_COUNT DCP shard files"

# Check metadata content
CKPT_STEP=$(python -c "import torch; m=torch.load('$STEP_DIR/training_state.pt', weights_only=False); print(m['step'])")
echo "  OK: Checkpoint at step $CKPT_STEP"

echo ""
echo "=== Phase 3: Resume from checkpoint ==="
RESUME_EXIT=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
    -m mini_trainer.train \
    --model-name-or-path "$MODEL" \
    --data-path "$DATA" \
    --batch-size 2 \
    --max-tokens-per-gpu 512 \
    --learning-rate 1e-4 \
    --output-dir "$OUTPUT_DIR/resumed" \
    --training-mode step \
    --max-steps 50 \
    --osft \
    --osft-unfreeze-rank-ratio 0.2 \
    --resume-from-full-state-checkpoint "$STEP_DIR" \
    > "$OUTPUT_DIR/resume.log" 2>&1 || RESUME_EXIT=$?
if [ $RESUME_EXIT -ne 0 ]; then
    echo "FAIL: Resume training exited with code $RESUME_EXIT"
    grep "Error\|Traceback" "$OUTPUT_DIR/resume.log" | head -5
    exit 1
fi

# Check that training continued from the checkpoint step
RESUMED_STEPS=$(grep -c '"step":' "$OUTPUT_DIR/resume.log" || echo 0)
FIRST_RESUMED_STEP=$(grep '"step":' "$OUTPUT_DIR/resume.log" | head -1 | python -c "import sys,re; print(re.search(r'\"step\":\s*(\d+)', sys.stdin.read()).group(1))" 2>/dev/null || echo 0)
echo "  Resumed training ran $RESUMED_STEPS steps, starting from step $FIRST_RESUMED_STEP"

if [ "$FIRST_RESUMED_STEP" -le "$CKPT_STEP" ]; then
    echo "FAIL: First resumed step ($FIRST_RESUMED_STEP) should be > checkpoint step ($CKPT_STEP)"
    exit 1
fi

echo ""
echo "========================================"
echo "SIGNAL TEST PASSED"
echo "  Checkpoint saved at step $CKPT_STEP"
echo "  Resumed from step $FIRST_RESUMED_STEP"
echo "  Training completed $RESUMED_STEPS more steps"
echo "========================================"
