# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini Trainer is a lightweight, high-performance training library for fine-tuning large language models up to 70B parameters on a single node. It provides reference implementations of emerging training algorithms, with Orthogonal Subspace Fine-Tuning (OSFT) as its flagship capability. The library uses FSDP2 for distributed training, Liger Kernels for memory efficiency, and numba-optimized batch packing.

## Development Commands

### Installation
```bash
uv pip install -e .                            # Base installation
uv pip install -e .[cuda] --no-build-isolation  # With GPU support (flash-attn, liger-kernel)
uv pip install -e .[dev]                        # Development tools (ruff, pytest, tox)
uv pip install -e .[test]                       # Test dependencies only
```

### Testing

Mini Trainer uses tox with GPU/non-GPU test separation:

```bash
# Non-GPU tests (CI default, no CUDA required)
tox -e py312-nogpu
tox -e py311-nogpu

# GPU tests (requires CUDA)
tox -e py312-gpu
tox -e py311-gpu

# With coverage
tox -e py312-nogpu-cov

# Direct pytest (after installing test deps)
pytest                                    # All tests
pytest -m "not gpu and not multi_gpu"     # Non-GPU only
pytest tests/gpu_tests/                   # GPU tests only
pytest -x                                 # Stop on first failure
```

Test markers: `slow`, `integration`, `unit`, `gpu`, `multi_gpu`

Regression tests live in `regression_tests/` (separate from the unit test suite in `tests/`).

### Code Quality
```bash
ruff check .          # Lint
ruff check --fix .    # Lint with auto-fix
ruff format .         # Format
ruff format --check . # Check formatting
```

Ruff is configured for line-length 120, targeting Python 3.11.

### Running Training
```bash
# Distributed training via torchrun
torchrun --nnodes=1 --nproc-per-node=8 -m mini_trainer.train \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./data.jsonl \
    --output-dir ./checkpoints \
    --batch-size 128 \
    --max-tokens-per-gpu 128000 \
    --learning-rate 5e-6

# With OSFT
torchrun ... -m mini_trainer.train --osft --osft-unfreeze-rank-ratio 0.25

# Pretraining mode
torchrun ... -m mini_trainer.train --block-size 4096
```

## Architecture

### Source Layout
```
src/mini_trainer/
  train.py                  # Main training loop (typer CLI entry point)
  api_train.py              # Programmatic API (run_training wrapper around torchrun)
  training_types.py         # TrainingArgs, TorchrunArgs, PretrainingConfig dataclasses
  setup_model_for_training.py  # Model loading, FSDP2 wrapping, optimizer setup
  fsdp2_lazy_init.py        # Lazy initialization for FSDP2 distributed setup
  osft_utils.py             # OSFT: SVD decomposition, subspace projection, gradient hooks
  batch_packer.py           # Numba-optimized LPT bin-packing for minibatches
  sampler.py                # Data loading with infinite sampling support
  batch_metrics.py          # Per-batch loss and token tracking
  full_state_checkpoint.py  # Signal-driven full-state checkpointing (model+optimizer+scheduler+RNG)
  none_reduction_losses.py  # Custom loss functions (no reduction, for per-token tracking)
  utils.py                  # Distributed helpers, logging, seeding
  vlm_utils.py              # Vision-language model utilities
  async_structured_logger.py  # Async JSONL metrics logger
  wandb_wrapper.py          # Weights & Biases integration
  mlflow_wrapper.py         # MLflow integration
```

### Key Design Patterns

- **CLI + Programmatic API**: `train.py` is the typer CLI; `api_train.py` wraps it as a subprocess via `torchrun` for programmatic use
- **FSDP2 Native**: Uses PyTorch's FullyShardedDataParallel v2 for distributed training (not DeepSpeed, not FSDP1)
- **Pre-tokenized Data**: Expects JSONL with `input_ids`, `labels`, and optional `len` fields. Data processing is handled externally (e.g., instructlab-training)
- **OSFT Subspace Projection**: Decomposes weight matrices via SVD, freezes important subspaces, trains in the orthogonal complement. Controlled by `--osft-unfreeze-rank-ratio`
- **Batch Packing**: Uses numba-JIT LPT algorithm to pack variable-length sequences into fixed-capacity minibatches, minimizing padding waste
- **Signal-Driven Checkpointing**: Catches SIGTERM/SIGINT/SIGUSR1 to save full training state (model, optimizer, scheduler, RNG) before shutdown

### Data Format
```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...], "len": 128}
```
Use `-100` in labels for tokens to ignore in loss computation. The `len` field is optional.

## CI

PR tests run via GitHub Actions (`.github/workflows/pr-tests.yml`):
- Python 3.11 and 3.12 non-GPU tests (required)
- Coverage upload to Codecov (required)
- Ruff lint and format checks (non-blocking, `continue-on-error: true`)

## Python Version

Requires Python >=3.11, <3.13. PyTorch >=2.6.
