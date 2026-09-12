#!/usr/bin/env bash
# GPU tests, run by .github/workflows/gpu-tests.yml inside a MiniCloud job:
# a throwaway copy of the CI account's `ci-base` workspace on the CI node,
# with this checkout at ~/src. The venv is built fresh from the PR's own
# dependencies. uv's wheel cache on the workspace makes rebuilds fast.
set -euo pipefail

echo "== $(date -u +%FT%TZ) $(git rev-parse --short HEAD) on $(hostname) =="
nvidia-smi -L

VENV=~/venvs/mini_trainer
rm -rf "$VENV"
uv venv -q --python 3.12 "$VENV"
source "$VENV/bin/activate"

uv pip install -e ".[test]"
uv pip install -e ".[cuda]" --no-build-isolation
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), 'flash_attn', flash_attn.__version__)"

export TESTING=true
pytest tests/gpu_tests -v --tb=short -p no:xdist \
  ${MINICLOUD_JOB_DIR:+--junitxml "$MINICLOUD_JOB_DIR/junit.xml"}
