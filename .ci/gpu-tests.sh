#!/usr/bin/env bash
# GPU tests, run by .github/workflows/gpu-tests.yml inside a MiniCloud job:
# a throwaway copy of the CI account's `ci-base` workspace on the CI node,
# with this checkout at ~/src and a venv at ~/venvs/mini_trainer that
# already holds torch and the [cuda] extras (flash-attn, liger, mamba-ssm).
# The install below only adds what this branch changed.
set -euo pipefail

echo "== $(date -u +%FT%TZ) $(git rev-parse --short HEAD) on $(hostname) =="
nvidia-smi -L

source ~/venvs/mini_trainer/bin/activate
uv pip install -e ".[cuda,test]" --no-build-isolation
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), 'flash_attn', flash_attn.__version__)"

export TESTING=true
pytest tests/gpu_tests -v --tb=short -p no:xdist

# probe: does a pull_request_target run report the required gpu-tests check on a PR (2026-09-11)
