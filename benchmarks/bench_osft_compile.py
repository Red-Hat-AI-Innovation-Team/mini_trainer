"""Benchmark: OSFT eager vs compiled forward/backward.

Profiles OSFT training steps with torch.profiler to identify graph breaks
and measure compile speedup.

Usage:
  # Eager baseline
  python bench_osft_compile.py

  # Compiled
  python bench_osft_compile.py --compile

  # With Chrome trace export
  python bench_osft_compile.py --compile --trace-dir benchmarks/traces

  # dynamo.explain report (graph break analysis)
  python bench_osft_compile.py --explain
"""

import argparse
import os
import time

os.environ["TESTING"] = "true"

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import LlamaConfig, LlamaForCausalLM

from mini_trainer.none_reduction_losses import hf_fixed_cross_entropy_none_reduction
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import patch_target_module


def create_tiny_llama(tmp_dir):
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(tmp_dir)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp_dir)
    return tmp_dir


def setup_dist():
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12399")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    patch_target_module(
        "transformers.loss.loss_utils.fixed_cross_entropy",
        hf_fixed_cross_entropy_none_reduction,
    )


def build_model(model_path, compile_model, osft=True, osft_rank_ratio=0.25):
    model = setup_model(
        model_name_or_path=model_path,
        use_liger_kernels=False,
        osft=osft,
        osft_rank_ratio=osft_rank_ratio if osft else None,
        local_rank=0,
    )
    model, optimizer, lr_scheduler = setup_training_components(
        model,
        learning_rate=1e-3,
        num_warmup_steps=0,
        lr_scheduler="constant",
        compile_model=compile_model,
    )
    return model, optimizer, lr_scheduler


def run_steps(model, optimizer, lr_scheduler, input_ids, labels, num_steps):
    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss.float().sum() / input_ids.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())
    return losses


def run_explain(model_path):
    """Run torch._dynamo.explain on an OSFT model to report graph breaks."""
    model, optimizer, lr_scheduler = build_model(model_path, compile_model=False, osft=True)

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
    labels = input_ids.clone()

    explanation = torch._dynamo.explain(model)(input_ids=input_ids, labels=labels)
    print("\n" + "=" * 80)
    print("torch._dynamo.explain report")
    print("=" * 80)
    print(explanation)
    print("=" * 80)


def run_profile(model_path, compile_model, trace_dir, num_warmup=3, num_active=5):
    mode_str = "compiled" if compile_model else "eager"
    osft_str = "osft"
    print(f"\nProfiling {osft_str} {mode_str}...")

    model, optimizer, lr_scheduler = build_model(model_path, compile_model=compile_model, osft=True)

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
    labels = input_ids.clone()

    # Warmup (includes compilation for compiled mode)
    print(f"  Warmup: {num_warmup} steps...")
    t0 = time.perf_counter()
    run_steps(model, optimizer, lr_scheduler, input_ids, labels, num_warmup)
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup done in {warmup_time:.2f}s")

    # Profiled steps
    print(f"  Profiling: {num_active} steps...")
    trace_path = None
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"osft_{mode_str}.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=0, warmup=1, active=num_active - 1, repeat=1),
    ) as prof:
        for _ in range(num_active):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss.float().sum() / input_ids.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            prof.step()

    torch.cuda.synchronize()

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"  Chrome trace: {trace_path}")

    print(f"\n  === {osft_str} {mode_str} — Top CUDA ops ===")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        )
    )

    # Wall-clock timing (separate from profiler)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_steps(model, optimizer, lr_scheduler, input_ids, labels, 10)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"\n  Wall-clock: 10 steps in {wall:.3f}s ({wall / 10 * 1000:.1f} ms/step)")

    return prof


def main():
    parser = argparse.ArgumentParser(description="OSFT compile benchmark")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--explain", action="store_true", help="Run torch._dynamo.explain")
    parser.add_argument("--trace-dir", type=str, default=None, help="Directory for Chrome traces")
    parser.add_argument("--both", action="store_true", help="Run both eager and compiled for comparison")
    args = parser.parse_args()

    import tempfile

    tmpdir = tempfile.mkdtemp()

    torch.manual_seed(42)
    setup_dist()
    model_path = create_tiny_llama(tmpdir)

    try:
        if args.explain:
            run_explain(model_path)
        elif args.both:
            run_profile(model_path, compile_model=False, trace_dir=args.trace_dir)
            torch._dynamo.reset()
            dist.destroy_process_group()
            setup_dist()
            run_profile(model_path, compile_model=True, trace_dir=args.trace_dir)
        else:
            run_profile(model_path, compile_model=args.compile, trace_dir=args.trace_dir)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
