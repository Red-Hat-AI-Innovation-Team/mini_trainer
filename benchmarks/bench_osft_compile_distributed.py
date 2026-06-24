"""Benchmark: OSFT eager vs compiled training step throughput.

Measures steady-state step time for OSFT training with and without
torch.compile on a real model (Llama-8B) under FSDP2.

Usage:
  # Run on 6 GPUs (adjust --nproc-per-node as needed)
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --nproc-per-node=6 \
      benchmarks/bench_osft_compile_distributed.py

  # Eager only
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --nproc-per-node=6 \
      benchmarks/bench_osft_compile_distributed.py --mode eager

  # Compiled only
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --nproc-per-node=6 \
      benchmarks/bench_osft_compile_distributed.py --mode compiled
"""

import argparse
import gc
import json
import os
import statistics
import time

os.environ["TESTING"] = "true"

import torch
import torch.distributed as dist

from mini_trainer.none_reduction_losses import hf_fixed_cross_entropy_none_reduction
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import log_rank_0, patch_target_module


def parse_args():
    parser = argparse.ArgumentParser(description="OSFT compile benchmark (distributed)")
    parser.add_argument(
        "--mode",
        choices=["both", "eager", "compiled"],
        default="both",
        help="Which mode(s) to benchmark",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="Model name or path")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048", help="Comma-separated sequence lengths")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps (excluded from timing)")
    parser.add_argument("--measure-steps", type=int, default=20, help="Steps to measure")
    parser.add_argument("--osft-rank-ratio", type=float, default=0.25, help="OSFT rank ratio")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write results JSON")
    return parser.parse_args()


def run_arm(
    model_name: str,
    compile_model: bool,
    seq_lens: list[int],
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    osft_rank_ratio: float,
    local_rank: int,
):
    label = "compiled" if compile_model else "eager"
    log_rank_0(f"\n{'='*70}")
    log_rank_0(f"  {label.upper()} ARM")
    log_rank_0(f"{'='*70}")

    if compile_model:
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    model = setup_model(
        model_name_or_path=model_name,
        use_liger_kernels=False,
        osft=True,
        osft_rank_ratio=osft_rank_ratio,
        local_rank=local_rank,
    )
    model, optimizer, lr_scheduler = setup_training_components(
        model,
        learning_rate=1e-5,
        num_warmup_steps=0,
        lr_scheduler="constant",
        compile_model=compile_model,
    )

    results_by_seqlen = {}

    for seq_len in seq_lens:
        log_rank_0(f"\n--- {label} | seq_len={seq_len} ---")

        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{local_rank}")
        labels = input_ids.clone()

        # Warmup — extra steps for compiled to amortize torch.compile
        actual_warmup = warmup_steps + (5 if compile_model else 0)
        log_rank_0(f"  Warmup: {actual_warmup} steps...")
        for _ in range(actual_warmup):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss.float().sum() / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

        torch.cuda.synchronize()
        dist.barrier()

        # Measure with CUDA events — no host sync between steps
        log_rank_0(f"  Measuring: {measure_steps} steps...")
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_steps)]

        for i in range(measure_steps):
            start_events[i].record()

            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss.float().sum() / batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            end_events[i].record()

        torch.cuda.synchronize()
        step_times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        dist.barrier()

        median_ms = statistics.median(step_times_ms)
        mean_ms = statistics.mean(step_times_ms)
        stdev_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
        p10 = sorted(step_times_ms)[max(0, len(step_times_ms) // 10)]
        p90 = sorted(step_times_ms)[min(len(step_times_ms) - 1, 9 * len(step_times_ms) // 10)]

        world_size = dist.get_world_size()
        tokens_per_step = batch_size * seq_len * world_size
        tokens_per_sec = tokens_per_step / (median_ms / 1000)

        results_by_seqlen[seq_len] = {
            "median_ms": round(median_ms, 2),
            "mean_ms": round(mean_ms, 2),
            "stdev_ms": round(stdev_ms, 2),
            "p10_ms": round(p10, 2),
            "p90_ms": round(p90, 2),
            "tokens_per_sec": round(tokens_per_sec, 0),
            "all_steps_ms": [round(t, 2) for t in step_times_ms],
        }

        log_rank_0(
            f"  Result: median={median_ms:.1f}ms  mean={mean_ms:.1f}ms  "
            f"stdev={stdev_ms:.1f}ms  p10={p10:.1f}ms  p90={p90:.1f}ms  "
            f"tok/s={tokens_per_sec:.0f}"
        )

    # Cleanup
    del model, optimizer, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    return results_by_seqlen


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    patch_target_module(
        "transformers.loss.loss_utils.fixed_cross_entropy",
        hf_fixed_cross_entropy_none_reduction,
    )

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    world_size = dist.get_world_size()

    log_rank_0(f"\nOSFT Compile Benchmark")
    log_rank_0(f"  Model: {args.model}")
    log_rank_0(f"  GPUs: {world_size}x {torch.cuda.get_device_name(local_rank)}")
    log_rank_0(f"  Seq lengths: {seq_lens}")
    log_rank_0(f"  Batch size: {args.batch_size}/GPU")
    log_rank_0(f"  OSFT rank ratio: {args.osft_rank_ratio}")
    log_rank_0(f"  Warmup: {args.warmup_steps} steps, Measure: {args.measure_steps} steps")

    all_results = {
        "config": {
            "model": args.model,
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(local_rank),
            "batch_size_per_gpu": args.batch_size,
            "osft_rank_ratio": args.osft_rank_ratio,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "pytorch_version": torch.__version__,
        }
    }

    modes = []
    if args.mode == "both":
        modes = ["eager", "compiled"]
    elif args.mode == "eager":
        modes = ["eager"]
    else:
        modes = ["compiled"]

    for mode in modes:
        compile_model = mode == "compiled"
        results = run_arm(
            model_name=args.model,
            compile_model=compile_model,
            seq_lens=seq_lens,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            osft_rank_ratio=args.osft_rank_ratio,
            local_rank=local_rank,
        )
        all_results[mode] = results

    # Summary table
    if len(modes) == 2 and dist.get_rank() == 0:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: OSFT eager vs compiled")
        print(f"  {args.model} | {world_size} GPUs | batch={args.batch_size}/GPU | rank_ratio={args.osft_rank_ratio}")
        print(f"{'='*70}")
        print(f"  {'seq_len':>8}  {'eager_ms':>10}  {'compiled_ms':>12}  {'speedup':>8}  {'tok/s eager':>12}  {'tok/s compiled':>15}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*15}")
        for sl in seq_lens:
            e = all_results["eager"][sl]
            c = all_results["compiled"][sl]
            speedup = e["median_ms"] / c["median_ms"]
            print(
                f"  {sl:>8}  {e['median_ms']:>10.1f}  {c['median_ms']:>12.1f}  {speedup:>7.2f}x  "
                f"{e['tokens_per_sec']:>12.0f}  {c['tokens_per_sec']:>15.0f}"
            )
        print(f"{'='*70}")

    if args.output_json and dist.get_rank() == 0:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
