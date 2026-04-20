"""
Test full-state checkpointing with all supported model architectures.

For each model x liger mode:
1. Run OSFT training with --on-demand-checkpointing
2. Write trigger file to /dev/shm after a few steps
3. Verify checkpoint was saved and training exited cleanly
4. Resume from checkpoint
5. Verify resumed training completes
6. Clean up artifacts

Skips models that OOM on available GPUs.

Usage:
    python regression_tests/test_checkpoint_all_models.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

# All models from model_validation.py
MODELS = {
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "qwen2": {
        "model_id": "qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "qwen3": {
        "model_id": "qwen/Qwen3-4B-Instruct-2507",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "granite": {
        "model_id": "ibm-granite/granite-3.1-8b-instruct",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "granite-moe": {
        "model_id": "ibm-granite/granite-4.0-h-tiny",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "phi4": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "gemma3": {
        "model_id": "google/gemma-3-4b-it",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "gemma3n": {
        "model_id": "google/gemma-3n-E4B-it",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "ministral": {
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "mistral3-vlm": {
        "model_id": "mistralai/Ministral-3-3B-Reasoning-2512",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "qwen3-vl": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "qwen3.5": {
        "model_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
    "nemotron": {
        "model_id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "trust_remote_code": True,
        "osft_target_patterns": "q_proj,k_proj,v_proj,o_proj",
    },
    "gpt-oss": {
        "model_id": "openai/gpt-oss-20b",
        "trust_remote_code": False,
        "osft_target_patterns": None,
    },
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_REPO_ROOT, ".test_artifacts", "messages_data.jsonl")
BASE_OUTPUT = os.path.join(_REPO_ROOT, ".test_artifacts", "runs")
TRIGGER_FILE = "/dev/shm/mini_trainer_checkpoint_trigger"
NUM_GPUS = 8


def clean_trigger():
    """Remove stale trigger file."""
    try:
        os.unlink(TRIGGER_FILE)
    except FileNotFoundError:
        pass


def run_osft_training(
    model_id,
    output_dir,
    use_liger,
    trust_remote_code,
    osft_target_patterns,
    on_demand=False,
    resume_path=None,
    max_steps=30,
):
    """Launch OSFT training via training_hub.osft() and return the process."""
    # Use training_hub's osft function via a subprocess python call
    # This handles data processing + torchrun automatically
    script = f"""
import os, sys
sys.path.insert(0, '/mnt/nvme3n1/workspace/osilkin/pr-repos/async-checkpointing/training_hub/src')
sys.path.insert(0, '/mnt/nvme3n1/workspace/osilkin/pr-repos/async-checkpointing/mini_trainer/src')
from training_hub import osft

osft(
    model_path="{model_id}",
    data_path="{DATA_PATH}",
    ckpt_output_dir="{output_dir}",
    unfreeze_rank_ratio=0.5,
    num_epochs=1,
    effective_batch_size=8,
    learning_rate=1e-4,
    max_seq_len=512,
    max_tokens_per_gpu=512,
    data_output_dir="{output_dir}/_data",
    warmup_steps=0,
    use_liger={use_liger},
    seed=42,
    lr_scheduler="cosine",
    checkpoint_at_epoch=False,
    save_final_checkpoint=False,
    trust_remote_code={trust_remote_code},
    nproc_per_node={NUM_GPUS},
    nnodes=1,
    node_rank=0,
    training_mode="step",
    max_steps={max_steps},
    on_demand_checkpointing={on_demand},
    {"resume_from_full_state_checkpoint='" + resume_path + "'," if resume_path else ""}
    {("osft_target_patterns='" + osft_target_patterns + "',") if osft_target_patterns else ""}
)
"""
    # Write stdout to a log file instead of PIPE to avoid blocking on
    # grandchild processes (torchrun workers) that inherit the pipe.
    log_path = os.path.join(output_dir, "subprocess.log")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc._log_fh = log_fh  # keep reference for cleanup
    proc._log_path = log_path
    return proc


def wait_for_steps(output_dir, min_steps=5, timeout=600):
    """Wait until training has logged at least min_steps metric lines."""
    start = time.time()
    while time.time() - start < timeout:
        # Look for metrics file
        for f in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
            if f.startswith("training_metrics_") and f.endswith(".jsonl"):
                path = os.path.join(output_dir, f)
                try:
                    with open(path) as fh:
                        lines = sum(1 for _ in fh)
                    if lines >= min_steps:
                        return True
                except OSError:
                    pass
        time.sleep(2)
    return False


def find_checkpoint_dir(output_dir):
    """Find the full-state checkpoint directory."""
    ckpt_base = os.path.join(output_dir, "full_state_checkpoints")
    if not os.path.isdir(ckpt_base):
        return None
    steps = [d for d in os.listdir(ckpt_base) if d.startswith("step_")]
    if not steps:
        return None
    # Return the latest
    steps.sort(key=lambda x: int(x.split("_")[1]))
    return os.path.join(ckpt_base, steps[-1])


def test_model(model_key, model_info, use_liger):
    """Test checkpoint save + resume for a single model/liger combo."""
    liger_str = "liger" if use_liger else "no-liger"
    tag = f"{model_key}_{liger_str}"
    output_dir = os.path.join(BASE_OUTPUT, tag)

    print(f"\n{'=' * 70}")
    print(f"  {tag}: {model_info['model_id']}")
    print(f"{'=' * 70}")

    # Clean up from any previous run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    clean_trigger()

    # Phase 1: Run training with on-demand checkpointing
    print(f"  [{tag}] Phase 1: Training with on-demand checkpointing...")
    proc = run_osft_training(
        model_id=model_info["model_id"],
        output_dir=output_dir,
        use_liger=use_liger,
        trust_remote_code=model_info["trust_remote_code"],
        osft_target_patterns=model_info.get("osft_target_patterns"),
        on_demand=True,
        max_steps=100,  # enough steps that we can trigger mid-training
    )

    # Wait for training to produce steps
    started = wait_for_steps(output_dir, min_steps=5, timeout=600)
    if not started:
        # Check if process died (OOM or other error)
        ret = proc.poll()
        if ret is not None:
            output = open(proc._log_path).read() if hasattr(proc, "_log_path") else ""
            if "CUDA out of memory" in output or "OutOfMemoryError" in output:
                print(f"  [{tag}] SKIP: OOM — model too large for {NUM_GPUS} GPUs")
                shutil.rmtree(output_dir, ignore_errors=True)
                return "skip_oom"
            else:
                print(f"  [{tag}] FAIL: Training crashed before producing steps")
                # Save log for debugging
                log_path = os.path.join(output_dir, "crash.log")
                with open(log_path, "w") as f:
                    f.write(output)
                print(f"    Log saved to {log_path}")
                print(f"    Last 5 lines: {output.strip().split(chr(10))[-5:]}")
                shutil.rmtree(output_dir, ignore_errors=True)
                return "fail"
        print(f"  [{tag}] FAIL: Timed out waiting for training to start")
        proc.kill()
        proc.wait()
        shutil.rmtree(output_dir, ignore_errors=True)
        return "fail"

    # Write trigger file
    print(f"  [{tag}] Writing trigger file...")
    with open(TRIGGER_FILE, "w") as f:
        pass

    # Wait for process to exit
    try:
        proc.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  [{tag}] FAIL: Training didn't exit after trigger (timeout)")
        proc.kill()
        proc.wait()
        shutil.rmtree(output_dir, ignore_errors=True)
        return "fail"

    proc._log_fh.close()
    output_phase1 = open(proc._log_path).read()

    # Check exit code
    if proc.returncode != 0:
        # Check for OOM
        if "CUDA out of memory" in output_phase1 or "OutOfMemoryError" in output_phase1:
            print(f"  [{tag}] SKIP: OOM during training")
            shutil.rmtree(output_dir, ignore_errors=True)
            return "skip_oom"
        print(f"  [{tag}] FAIL: Training exited with code {proc.returncode}")
        print(f"    Last 3 lines: {output_phase1.strip().split(chr(10))[-3:]}")
        shutil.rmtree(output_dir, ignore_errors=True)
        return "fail"

    # Find checkpoint
    ckpt_dir = find_checkpoint_dir(output_dir)
    if ckpt_dir is None:
        print(f"  [{tag}] FAIL: No checkpoint directory found")
        shutil.rmtree(output_dir, ignore_errors=True)
        return "fail"

    print(f"  [{tag}] Checkpoint saved at {os.path.basename(ckpt_dir)}")

    # Phase 2: Resume from checkpoint
    print(f"  [{tag}] Phase 2: Resuming from checkpoint...")
    resume_dir = os.path.join(BASE_OUTPUT, f"{tag}_resumed")
    if os.path.exists(resume_dir):
        shutil.rmtree(resume_dir)
    os.makedirs(resume_dir, exist_ok=True)
    clean_trigger()

    proc2 = run_osft_training(
        model_id=model_info["model_id"],
        output_dir=resume_dir,
        use_liger=use_liger,
        trust_remote_code=model_info["trust_remote_code"],
        osft_target_patterns=model_info.get("osft_target_patterns"),
        on_demand=False,
        resume_path=ckpt_dir,
        max_steps=30,  # just needs to run a few more steps
    )

    try:
        proc2.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"  [{tag}] FAIL: Resume timed out")
        proc2.kill()
        proc2.wait()
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(resume_dir, ignore_errors=True)
        return "fail"

    proc2._log_fh.close()
    output_phase2 = open(proc2._log_path).read()

    if proc2.returncode != 0:
        if "CUDA out of memory" in output_phase2 or "OutOfMemoryError" in output_phase2:
            print(f"  [{tag}] SKIP: OOM during resume")
            shutil.rmtree(output_dir, ignore_errors=True)
            shutil.rmtree(resume_dir, ignore_errors=True)
            return "skip_oom"
        print(f"  [{tag}] FAIL: Resume exited with code {proc2.returncode}")
        print(f"    Last 3 lines: {output_phase2.strip().split(chr(10))[-3:]}")
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(resume_dir, ignore_errors=True)
        return "fail"

    print(f"  [{tag}] PASS")

    # Clean up
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(resume_dir, ignore_errors=True)
    return "pass"


def main():
    print("Full-State Checkpoint Validation — All Model Architectures")
    print(f"GPUs: {NUM_GPUS}, Data: {DATA_PATH}")
    print()

    results = {}
    for model_key, model_info in MODELS.items():
        for use_liger in [False, True]:
            liger_str = "liger" if use_liger else "no-liger"
            tag = f"{model_key}_{liger_str}"
            result = test_model(model_key, model_info, use_liger)
            results[tag] = result

    # Summary
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    passes = sum(1 for v in results.values() if v == "pass")
    fails = sum(1 for v in results.values() if v == "fail")
    skips = sum(1 for v in results.values() if v.startswith("skip"))
    print(f"  PASS: {passes}  FAIL: {fails}  SKIP: {skips}  TOTAL: {len(results)}")
    print()
    for tag, result in results.items():
        icon = {"pass": "OK", "fail": "FAIL", "skip_oom": "SKIP(OOM)"}
        print(f"  {icon.get(result, result):>10}  {tag}")

    # Clean up base output if empty
    if os.path.exists(BASE_OUTPUT) and not os.listdir(BASE_OUTPUT):
        shutil.rmtree(BASE_OUTPUT)

    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
