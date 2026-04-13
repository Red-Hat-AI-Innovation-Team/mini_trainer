"""
Full-state checkpoint fidelity test.

Proves that the optimization trajectory after checkpoint/resume is
bit-identical to a continuous run.

Run A (baseline):   Train 20 steps, record metrics at each step.
Run B (checkpoint):  Train 10 steps, checkpoint, restart, train 10 more steps.
Compare:             Steps 11-20 must be identical between A and B.

Usage:
    # Run A: baseline (20 steps straight through)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
        regression_tests/test_full_state_checkpoint_fidelity.py \
        --mode baseline --total-steps 20 --output-dir /tmp/fidelity_test

    # Run B part 1: train 10 steps then checkpoint
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
        regression_tests/test_full_state_checkpoint_fidelity.py \
        --mode checkpoint --checkpoint-at-step 10 --total-steps 20 \
        --output-dir /tmp/fidelity_test

    # Run B part 2: resume from checkpoint, train 10 more steps
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
        regression_tests/test_full_state_checkpoint_fidelity.py \
        --mode resume --total-steps 20 --output-dir /tmp/fidelity_test

    # Compare trajectories
    python regression_tests/test_full_state_checkpoint_fidelity.py \
        --mode compare --output-dir /tmp/fidelity_test
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def hash_parameters(model: torch.nn.Module) -> str:
    """Create a deterministic hash of all trainable model parameters.

    Only hashes local shards to avoid expensive all-gather operations.
    """
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        h.update(name.encode())
        # Use local tensor for DTensor to avoid all-gather
        p = param.detach()
        if hasattr(p, "_local_tensor"):
            p = p._local_tensor
        h.update(p.float().cpu().numpy().tobytes())
    return h.hexdigest()


def record_step_metrics(step, loss, grad_norm, lr, param_hash):
    return {
        "step": step,
        "loss": loss,
        "grad_norm": grad_norm,
        "lr": lr,
        "param_hash": param_hash,
    }


def run_training(args, checkpoint_at_step=None, resume_checkpoint=None):
    """Run training and record per-step metrics."""
    from mini_trainer.full_state_checkpoint import FullStateCheckpointer
    from mini_trainer.sampler import get_data_loader
    from mini_trainer.setup_model_for_training import setup_model, setup_training_components
    from mini_trainer.utils import init_distributed_environment, set_seed

    # Use inline gradient step to avoid validate_training_state dtype checks
    def take_gradient_step(model, optimizer, lr_scheduler):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    init_distributed_environment()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_seed(42)

    # Setup model
    model = setup_model(
        model_name_or_path=args.model_name_or_path,
        osft=True,
        osft_rank_ratio=0.8,  # unfreeze 20%
        use_liger_kernels=False,
        train_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    model, optimizer, lr_scheduler = setup_training_components(
        model=model,
        learning_rate=1e-4,
        num_warmup_steps=2,
        lr_scheduler="cosine",
        num_training_steps=args.total_steps,
        resume_from_checkpoint=resume_checkpoint,
    )

    data_loader, _ = get_data_loader(
        data_path=args.data_path,
        batch_size=2,
        max_tokens_per_gpu=512,
        seed=42,
    )

    device = next(model.parameters()).device
    metrics = []

    # If resuming, load optimizer + RNG + metadata
    start_step = 0
    start_epoch = 0
    if resume_checkpoint:
        meta = FullStateCheckpointer.load_metadata(resume_checkpoint)
        start_step = meta["step"]
        start_epoch = meta["epoch"]
        lr_scheduler.load_state_dict(meta["lr_scheduler_state"])
        FullStateCheckpointer.load_rng_states(resume_checkpoint, rank)

        # Load optimizer state from DCP (same approach as production code)
        import torch.distributed.checkpoint as dcp_module

        optim_state = optimizer.state_dict()
        dcp_module.load({"optimizer": optim_state}, checkpoint_id=str(Path(resume_checkpoint) / "distributed"))
        optimizer.load_state_dict(optim_state)

        data_loader.sampler.set_epoch(meta["sampler_epoch"])

    model.train()
    step = 0
    for epoch in range(start_epoch, 100):  # enough to cover total_steps
        data_loader.sampler.set_epoch(epoch)
        for batch in data_loader:
            # Skip batches if resuming
            if step < start_step:
                step += 1
                continue

            for mb in batch:
                model_inputs = {
                    "input_ids": mb["input_ids"].to(device),
                    "labels": mb["labels"].to(device),
                }
                if (pos_ids := mb.get("position_ids")) is not None:
                    model_inputs["position_ids"] = pos_ids.to(device)
                if (attn_mask := mb.get("attention_mask")) is not None:
                    model_inputs["attention_mask"] = attn_mask.to(device)

                output = model(**model_inputs)
                loss = output.loss.float().sum()
                batch_tokens = mb["batch_num_loss_counted_tokens"]
                loss = (loss / batch_tokens) * world_size
                loss.backward()
                torch.cuda.empty_cache()

            step += 1
            current_lr = lr_scheduler.get_last_lr()[0]
            grad_norm = take_gradient_step(model, optimizer, lr_scheduler)

            if rank == 0:
                metrics.append(
                    record_step_metrics(
                        step=step,
                        loss=loss.detach().cpu().item(),
                        grad_norm=grad_norm.item(),
                        lr=current_lr,
                        param_hash=hash_parameters(model),
                    )
                )
                print(
                    f"  step={step} loss={loss.detach().cpu().item():.6f} grad_norm={grad_norm.item():.6f} lr={current_lr:.8f}"
                )

            dist.barrier()

            # Checkpoint if requested
            if checkpoint_at_step and step == checkpoint_at_step:
                checkpointer = FullStateCheckpointer(
                    checkpoint_dir=os.path.join(args.output_dir, "full_state_checkpoints"),
                    rank=rank,
                    world_size=world_size,
                )
                checkpointer.save(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    training_state={
                        "step": step,
                        "epoch": epoch,
                        "total_samples_accumulated": step * 2,
                        "total_tokens_processed": 0,
                        "last_validation_loss": None,
                    },
                    checkpointer_state={
                        "last_saved_samples": 0,
                        "last_frequency_saved_samples": 0,
                        "best_val_loss": None,
                    },
                    data_loader=data_loader,
                    device=device,
                )
                if rank == 0:
                    torch.save(metrics, os.path.join(args.output_dir, "trajectory_first_half.pt"))
                    print(f"Checkpoint saved at step {step}. Exiting.")
                dist.barrier()
                dist.destroy_process_group()
                os._exit(0)

            if step >= args.total_steps:
                break
        if step >= args.total_steps:
            break

    # Save metrics
    if rank == 0:
        suffix = "baseline" if not resume_checkpoint else "resumed"
        torch.save(metrics, os.path.join(args.output_dir, f"trajectory_{suffix}.pt"))
        print(f"Saved {len(metrics)} step metrics as trajectory_{suffix}.pt")

    dist.destroy_process_group()


def compare_trajectories(output_dir):
    """Compare baseline and resumed trajectories step by step."""
    baseline = torch.load(os.path.join(output_dir, "trajectory_baseline.pt"), weights_only=False)
    first_half = torch.load(os.path.join(output_dir, "trajectory_first_half.pt"), weights_only=False)
    resumed = torch.load(os.path.join(output_dir, "trajectory_resumed.pt"), weights_only=False)

    combined = first_half + resumed

    if len(baseline) != len(combined):
        print(f"FIDELITY TEST FAILED: trajectory length mismatch baseline={len(baseline)}, combined={len(combined)}")
        sys.exit(1)

    mismatches = []
    for b, c in zip(baseline, combined):
        step = b["step"]
        if b["loss"] != c["loss"]:
            mismatches.append(f"Step {step}: loss {b['loss']} != {c['loss']}")
        if b["grad_norm"] != c["grad_norm"]:
            mismatches.append(f"Step {step}: grad_norm {b['grad_norm']} != {c['grad_norm']}")
        if b["param_hash"] != c["param_hash"]:
            mismatches.append(f"Step {step}: param_hash mismatch")
        if b["lr"] != c["lr"]:
            mismatches.append(f"Step {step}: lr {b['lr']} != {c['lr']}")

    if mismatches:
        print("FIDELITY TEST FAILED")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)
    else:
        print(f"FIDELITY TEST PASSED: all {len(baseline)} steps match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "checkpoint", "resume", "compare"], required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", default="test.jsonl")
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--checkpoint-at-step", type=int, default=10)
    parser.add_argument("--output-dir", default="/tmp/fidelity_test")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "baseline":
        run_training(args)
    elif args.mode == "checkpoint":
        run_training(args, checkpoint_at_step=args.checkpoint_at_step)
    elif args.mode == "resume":
        ckpt_dir = os.path.join(args.output_dir, "full_state_checkpoints", f"step_{args.checkpoint_at_step}")
        run_training(args, resume_checkpoint=ckpt_dir)
    elif args.mode == "compare":
        compare_trajectories(args.output_dir)
