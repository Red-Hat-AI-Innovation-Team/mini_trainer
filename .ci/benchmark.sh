#!/usr/bin/env bash
# The torch.compile benchmark, run by .github/workflows/benchmark.yml
# inside a MiniCloud job: a throwaway copy of the CI account's `ci-base`
# workspace with this checkout (main) at ~/bench/main-src and a venv at
# ~/venvs/mini_trainer that already holds torch, the [cuda] extras and the
# C toolchain inductor needs. Three configs, one after another:
#
#   baseline     main, eager
#   pr-eager     the pull request's head, eager
#   pr-compiled  the pull request's head, --compile-model
#
# Every config trains the same model on the same synthetic data for the
# same number of steps under TESTING=true (SDPA for all three). Each run's
# training_metrics_0.jsonl and its log go into results.json in the job's
# folder ($MINICLOUD_JOB_DIR, kept with the job's record), and
# benchmark_verdict.py renders comment.md from it.
#
# Knobs (environment, set by the workflow):
#   CI_REPO      owner/name (the clone the PR is fetched from)
#   BENCH_PR     the pull request number (required)
#   BENCH_MODE   sft (default) or osft (fp32, the OSFT flags)
#   BENCH_MODEL  model id (default Qwen/Qwen2.5-0.5B-Instruct)
#   BENCH_STEPS  training steps per config (default 20; steps 1-3 warm up)
#   BENCH_GPUS   processes per node for torchrun (default 1)
set -euo pipefail

: "${BENCH_PR:?BENCH_PR is the pull request number}"
: "${CI_REPO:?CI_REPO is owner/name}"
MODE=${BENCH_MODE:-sft}
MODEL=${BENCH_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STEPS=${BENCH_STEPS:-20}
GPUS=${BENCH_GPUS:-1}
OUT=${MINICLOUD_JOB_DIR:-$HOME/bench/out}
WORK=$HOME/bench
MAIN=$(cd "$(dirname "$0")/.." && pwd)
PR=$WORK/pr-src

echo "== $(date -u +%FT%TZ) benchmark PR #$BENCH_PR mode=$MODE model=$MODEL steps=$STEPS gpus=$GPUS on $(hostname) =="
nvidia-smi -L
mkdir -p "$OUT" "$WORK"

echo "== sources"
rm -rf "$PR"
git clone -q "$MAIN" "$PR"
git -C "$PR" fetch -q "https://github.com/$CI_REPO" "pull/$BENCH_PR/head:prhead"
git -C "$PR" checkout -q prhead
echo "main $(git -C "$MAIN" rev-parse --short HEAD)  pr $(git -C "$PR" rev-parse --short HEAD)"

source ~/venvs/mini_trainer/bin/activate
export TESTING=true                       # SDPA: the same attention for all three
export TORCHINDUCTOR_CACHE_DIR=$WORK/inductor-cache
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

echo "== data: 100 x 512 random tokens from the model's vocab"
python - "$MODEL" "$WORK/data.jsonl" <<'PY'
import json, random, sys
from transformers import AutoConfig, AutoTokenizer
model, out = sys.argv[1], sys.argv[2]
AutoTokenizer.from_pretrained(model)          # warms the cache for the runs
vocab = AutoConfig.from_pretrained(model).vocab_size
random.seed(42)
with open(out, "w") as f:
    for _ in range(100):
        ids = [random.randrange(1, vocab) for _ in range(512)]
        f.write(json.dumps({"input_ids": ids, "labels": list(ids)}) + "\n")
print(f"vocab {vocab}")
PY

DTYPE=bfloat16
EXTRA=()
if [ "$MODE" = osft ]; then
  DTYPE=float32
  EXTRA=(--osft --osft-unfreeze-rank-ratio 0.25)
fi

run_cfg() {  # name  source-dir  extra flags...
  local name=$1 src=$2; shift 2
  echo "== $name: $(git -C "$src" rev-parse --short HEAD) $*"
  uv pip install -q -e "$src" --no-build-isolation
  local rc=0
  (cd "$src" && timeout 3600 torchrun --nproc-per-node="$GPUS" --nnodes=1 src/mini_trainer/train.py \
      --model-name-or-path "$MODEL" \
      --data-path "$WORK/data.jsonl" \
      --batch-size 4 \
      --max-tokens-per-gpu 4096 \
      --learning-rate 2e-5 \
      --train-dtype "$DTYPE" \
      --training-mode step \
      --max-steps "$STEPS" \
      --seed 42 \
      --output-dir "$WORK/out-$name" \
      --num-warmup-steps 0 \
      "${EXTRA[@]}" "$@") > "$OUT/bench-$name.log" 2>&1 || rc=$?
  echo "$rc" > "$WORK/rc-$name"
  echo "   rc=$rc"
}

run_cfg baseline    "$MAIN"
run_cfg pr-eager    "$PR"
run_cfg pr-compiled "$PR" --compile-model

echo "== results"
MAIN_SRC="$MAIN" python - "$WORK" "$OUT" "$MODE" "$MODEL" "$STEPS" "$GPUS" "$BENCH_PR" <<'PY'
import json, os, subprocess, sys
work, out, mode, model, steps, gpus, pr = sys.argv[1:8]
sha = lambda d: subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
res = {"pr": int(pr), "mode": mode, "model": model, "steps": int(steps), "gpus": int(gpus),
       "shas": {"main": sha(os.environ["MAIN_SRC"]), "pr": sha(f"{work}/pr-src")}}
for name in ("baseline", "pr-eager", "pr-compiled"):
    rc = int(open(f"{work}/rc-{name}").read().strip())
    metrics = []
    for cand in (f"{work}/out-{name}/training_metrics_0.jsonl", f"{work}/out-{name}/checkpoints/training_metrics_0.jsonl"):
        if os.path.exists(cand):
            for line in open(cand):
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except ValueError:
                        pass
            break
    res[name] = {"rc": rc, "metrics": metrics, "log": f"bench-{name}.log"}
json.dump(res, open(f"{out}/results.json", "w"))
print("results.json:", {k: (v["rc"], len(v["metrics"])) for k, v in res.items() if isinstance(v, dict) and "rc" in v})
PY
python "$MAIN/.ci/benchmark_verdict.py" "$OUT/results.json" > "$OUT/comment.md"
echo "== comment.md"; cat "$OUT/comment.md"
