"""Benchmark: Gram vs factored V projection under FSDP2-style sharding.

Measures the V projection step of project_gradient_to_orthogonal_space()
with Llama-8B shapes.  Compares three modes:

  Gram:     all-reduce (M, M) Gram matrix, then dV -= dV @ G
  Factored: all-gather (k_high, M) V_high, then dV -= (dV @ V^T) @ V
  Cached:   reuse V_high from prior step (no comm), same matmuls

Usage:
  python bench_v_proj.py           # requires 2 GPUs
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def bench_target(name, k_high, k_low, M, P, dev, n_iters=100):
    """Benchmark one OSFT target shape.  Returns dict of timings."""
    if k_high % P != 0 or k_low % P != 0:
        raise ValueError(f"k_high={k_high} and k_low={k_low} must both be divisible by P={P}")
    local_k_high = k_high // P
    local_k_low = k_low // P

    # V_high has orthonormal rows (from SVD) — create via QR in float32
    V_full_f32 = torch.linalg.qr(torch.randn(M, k_high, device=dev))[0].T[:k_high]  # (k_high, M) orthonormal rows
    V_full = V_full_f32.to(torch.bfloat16)

    # Each rank's shard of V_high
    local_V = V_full[rank * local_k_high : (rank + 1) * local_k_high].contiguous()
    local_dV = torch.randn(local_k_low, M, device=dev, dtype=torch.bfloat16)
    dV_save = local_dV.clone()

    # Pre-allocate for all-gather
    V_ag = torch.empty_like(V_full)

    # Warmup
    for _ in range(10):
        local_dV.copy_(dV_save)
        G = torch.mm(local_V.T, local_V)
        dist.all_reduce(G)
        local_dV.add_(torch.mm(local_dV, G), alpha=-1.0)

        local_dV.copy_(dV_save)
        dist.all_gather_into_tensor(V_ag, local_V)
        coeff = torch.mm(local_dV, V_ag.T)
        local_dV.addmm_(coeff, V_ag, alpha=-1.0)
    torch.cuda.synchronize()
    dist.barrier()

    # --- Gram form: all-reduce (M, M), then dV -= dV @ G ---
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        local_dV.copy_(dV_save)
        G = torch.mm(local_V.T, local_V)
        dist.all_reduce(G)
        local_dV.add_(torch.mm(local_dV, G), alpha=-1.0)
    torch.cuda.synchronize()
    gram_ms = (time.perf_counter() - t0) / n_iters * 1000

    # --- Factored form: all-gather (k_high, M), then two matmuls ---
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        local_dV.copy_(dV_save)
        dist.all_gather_into_tensor(V_ag, local_V)
        coeff = torch.mm(local_dV, V_ag.T)
        local_dV.addmm_(coeff, V_ag, alpha=-1.0)
    torch.cuda.synchronize()
    fact_ms = (time.perf_counter() - t0) / n_iters * 1000

    # --- Cached: no comm, just matmuls ---
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        local_dV.copy_(dV_save)
        coeff = torch.mm(local_dV, V_ag.T)
        local_dV.addmm_(coeff, V_ag, alpha=-1.0)
    torch.cuda.synchronize()
    cached_ms = (time.perf_counter() - t0) / n_iters * 1000

    # Correctness: verify in float32 (bf16 accumulation differs by ~5%)
    dV_f32 = dV_save.float()
    V_f32 = V_full_f32
    gram_f32 = dV_f32 - dV_f32 @ (V_f32.T @ V_f32)
    fact_f32 = dV_f32 - (dV_f32 @ V_f32.T) @ V_f32
    max_diff = (fact_f32 - gram_f32).abs().max().item()

    return {
        "name": name,
        "k_high": k_high,
        "M": M,
        "gram_bytes": M * M * 2,
        "fact_bytes": k_high * M * 2,
        "gram_ms": gram_ms,
        "fact_ms": fact_ms,
        "cached_ms": cached_ms,
        "max_diff": max_diff,
    }


# Global for shard indexing
rank = 0


def run(rank_, world_size):
    global rank
    rank = rank_
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT="29500", RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    dev = f"cuda:{rank}"

    # Llama-8B shapes, URR=0.5
    targets = [
        # name,       k_high, k_low, M
        ("down_proj", 2048, 2048, 14336),  # (4096, 14336) → ratio = 7x
        ("q_proj", 2048, 2048, 4096),  # (4096, 4096)  → ratio = 2x
    ]

    results = []
    for name, k_high, k_low, M in targets:
        r = bench_target(name, k_high, k_low, M, world_size, dev)
        results.append(r)

    if rank == 0:
        gpu = torch.cuda.get_device_name(0)
        print(f"V projection benchmark — {world_size}× {gpu}, Llama-8B shapes, bf16")
        print("=" * 70)
        print()
        for r in results:
            ratio = r["M"] / r["k_high"]
            print(f"  {r['name']:10s}  V_high ({r['k_high']}, {r['M']})    M/k_high = {ratio:.0f}x")
            print(f"  {'':10s}  Gram        Factored    Cached")
            print(f"  {'':10s}  ----------  ----------  ----------")
            print(f"  {'comm':10s}  all-reduce  all-gather  none")
            print(f"  {'bytes':10s}  {r['gram_bytes'] / 1e6:>7.0f} MB  {r['fact_bytes'] / 1e6:>7.0f} MB  {0:>7.0f} MB")
            print(f"  {'time':10s}  {r['gram_ms']:>7.2f} ms  {r['fact_ms']:>7.2f} ms  {r['cached_ms']:>7.2f} ms")
            print(
                f"  {'speedup':10s}  {'—':>10s}  {r['gram_ms'] / r['fact_ms']:>7.1f}x    {r['gram_ms'] / r['cached_ms']:>7.1f}x"
            )
            ok = "ok" if r["max_diff"] < 1e-4 else "FAIL"
            print(f"  {'correct':10s}  —           f32 max diff = {r['max_diff']:.1e} ({ok})")
            print()

        # Aggregate: 32 layers x 7 targets (4 q/k/v/o + 2 gate/up + 1 down)
        sq = next(r for r in results if r["name"] == "q_proj")
        dp = next(r for r in results if r["name"] == "down_proj")
        gram_tot = (6 * sq["gram_ms"] + dp["gram_ms"]) * 32
        fact_tot = (6 * sq["fact_ms"] + dp["fact_ms"]) * 32
        cached_tot = (6 * sq["cached_ms"] + dp["cached_ms"]) * 32
        print("  Aggregate (32 layers x 7 targets = 224):")
        print(f"  {'':10s}  Gram        Factored    Cached")
        print(f"  {'total':10s}  {gram_tot:>7.0f} ms  {fact_tot:>7.0f} ms  {cached_tot:>7.0f} ms")
        print(f"  {'speedup':10s}  {'—':>10s}  {gram_tot / fact_tot:>7.1f}x    {gram_tot / cached_tot:>7.1f}x")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2, join=True)
