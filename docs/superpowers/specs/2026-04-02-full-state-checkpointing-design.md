# Full-State On-Demand Checkpointing for mini_trainer

## Problem

When training OSFT models in Kubernetes/OpenShift/SLURM environments, pods can be preempted at any time. Currently, mini_trainer only saves model weights (reconstructed to dense format) at epoch/sample boundaries. There is no way to resume training from the exact mid-training state, meaning all progress since the last checkpoint is lost.

## Goal

Implement signal-driven full-state checkpointing that:
1. Catches termination signals and saves complete training state before exit
2. Uses DCP sharded saves (each rank saves its shard, no gathering to rank 0)
3. Preserves the exact optimization trajectory on resume: `f(x1) -> save -> load -> f(x2)` must be bit-identical to `f(x1) -> f(x2)`
4. Saves the OSFT model in decomposed format (SVD factors, not reconstructed dense weights)

## Architecture

### New module: `src/mini_trainer/full_state_checkpoint.py`

Contains `FullStateCheckpointer` class with:
- `install_signal_handlers()` — registers handlers for 7 signals
- `should_save(device)` — polls trigger file + `all_reduce(MAX)` consensus
- `save(...)` — DCP sharded save + per-rank RNG + rank-0 metadata
- `load(...)` — DCP sharded load + state restoration (static/classmethod)
- `cleanup()` — removes trigger file

### Signals handled

SIGTERM, SIGINT, SIGUSR1, SIGUSR2, SIGXCPU, SIGHUP, SIGQUIT — covers Kubernetes, SLURM, PBS, LSF, and interactive use.

### Trigger mechanism

Signal handler writes a trigger file to `/dev/shm/mini_trainer_checkpoint_trigger`. Workers poll this file at batch boundaries (after `take_gradient_step()`). `all_reduce(MAX)` ensures all ranks agree. After save, trigger file is removed and processes exit cleanly.

### Checkpoint contents

**Sharded via `dcp.save()` (all ranks, no gathering):**
- Model state dict — OSFT factors (U/S/V_high, U/S/V_low) in decomposed form
- Optimizer state dict — AdamW momentum/variance, sharded to match model

**Per-rank via `torch.save()` (each rank saves its own):**
- RNG states: torch CPU, torch CUDA, python random, numpy

**Global via `torch.save()` (rank 0 only):**
- Training counters: step, epoch, total_samples_accumulated, total_tokens_processed
- last_validation_loss
- LR scheduler state dict
- Checkpointer state: last_saved_samples, last_frequency_saved_samples, best_val_loss
- Sampler epoch + accumulated samples (for batch-skipping on resume)
- OSFT config dict (sanity check on resume)

### Directory layout

```
<output_dir>/full_state_checkpoints/step_<N>/
  .metadata, __*_*.distcp     (DCP shards)
  rng_state_rank_*.pt          (per-rank RNG)
  training_state.pt            (global metadata, rank 0)
```

### Resume flow

On resume (`--resume-from-full-state-checkpoint <path>`):

1. `setup_model()` runs normally — loads model from pretrained on rank 0, applies all monkey-patches (loss function, Liger kernels, Mamba kernels, OSFT forward overrides), creates model structure on all ranks
2. `prepare_model_for_fsdp2()` — creates OSFT param structure with correct shapes on meta device
3. `wrap_fsdp2()` — FSDP2 shards the structure
4. `finalize_model_initialization()` — **skips `compute_distributed_svd()`**, instead uses `dcp.load()` to fill in parameter values from checkpoint
5. Create optimizer with OSFT wrapping — normal path
6. Load optimizer state from DCP checkpoint
7. Restore LR scheduler state, RNG states, training counters, checkpointer state
8. Resume training loop from saved step/epoch, skip already-processed batches

Key: SVD is NOT recomputed. The model structure (shapes, forward overrides, parameter registrations) is set up by the existing init pipeline. DCP load fills in the actual values.

### Monkey-patches that survive naturally

All monkey-patches are applied by the existing `setup_model()` / `setup_training_components()` pipeline, which runs before DCP load:
- Loss function patches (HF fixed_cross_entropy -> none_reduction)
- Liger kernel patches (via `_apply_liger_kernels_if_requested()`)
- Mamba-SSM kernel patches
- OSFT forward overrides (via `_pre_fsdp2_wrap_initialize_lazy_osft()`)
- Optimizer step wrapping (via `optim_wrapper()`)
- Config assignments (use_cache=False, torch_dtype, etc.)

### Training loop integration

One polling point per batch, after `take_gradient_step()`:

```python
if full_state_checkpointer and full_state_checkpointer.should_save(device):
    full_state_checkpointer.save(...)
    full_state_checkpointer.cleanup()
    sys.exit(0)
```

### Config additions

In `TrainingArgs` (training_types.py):
- `on_demand_checkpointing: bool = False`
- `resume_from_full_state_checkpoint: str | None = None`

### Separate from existing checkpoints

Full-state checkpoints are independent of the existing epoch/min_samples/best_val_loss checkpoint system. They are opaque resume tokens for this system only.

### Same-topology assumption

Initial implementation assumes same world size on resume. DCP's sharded format supports resharding, making future support straightforward.

## Test plan

### Test 1: Unit tests (pytest, no GPU)
- Signal handler writes trigger file
- Trigger detection and cleanup
- Metadata serialization roundtrip
- Config field parsing

### Test 2: Single-process save/load roundtrip (pytest, single GPU)
- Tiny inline model with OSFT
- Save and load, assert bit-identical model/optimizer/scheduler/RNG state

### Test 3: Optimization trajectory fidelity (torchrun, multi-GPU) — CRITICAL
- Qwen2.5-0.5B-Instruct, OSFT mode
- Run A: 20 steps baseline, record loss/grad_norm/param_hash/lr per step
- Run B: 10 steps -> checkpoint -> resume -> 10 more steps, record same metrics
- Assert steps 11-20 are identical between Run A and Run B

### Test 4: Multi-node simulation (torchrun, 2 NUMA domains)
- Same as Test 3 but with 2 simulated nodes: GPUs 0-3 (node 0) and GPUs 4-7 (node 1)
- Validates DCP sharded save/load across node boundaries

### Test 5: Signal-triggered end-to-end
- Launch training with --on-demand-checkpointing
- Send SIGUSR1 after a few steps
- Verify checkpoint directory structure, clean exit, and successful resume

### Test 6: Edge cases
- Checkpoint at step 1
- Checkpoint at epoch boundary
- SFT mode (non-OSFT, no SVD factors)
