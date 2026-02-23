"""
FSDP1 Model Wrapping Smoke Test

Minimal script to load a model, wrap it with FSDP1, and test forward/backward pass.
Vendors necessary components from accelerate without creating dependencies.

Usage:
    torchrun --nnodes=1 --nproc-per-node=8 fsdp1_dummy_script.py
"""

import functools
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def log_rank_0(msg: str):
    """Log only from rank 0."""
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"[Rank 0] {msg}")


def init_distributed():
    """Initialize distributed environment."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    dist.barrier()


def get_module_class_from_name(model: torch.nn.Module, module_name: str):
    """Get module class by name from model."""
    if model.__class__.__name__ == module_name:
        return model.__class__

    for child_module in model.children():
        module_class = get_module_class_from_name(child_module, module_name)
        if module_class is not None:
            return module_class
    return None


def wrap_model_with_fsdp1(model: torch.nn.Module) -> FSDP:
    """Wrap model with FSDP1 using minimal configuration."""
    log_rank_0("Wrapping model with FSDP1...")

    # Get transformer layer class for auto-wrap
    auto_wrap_policy = None
    if hasattr(model, "_no_split_modules") and model._no_split_modules:
        layer_name = next(iter(model._no_split_modules))
        layer_cls = get_module_class_from_name(model, layer_name)
        if layer_cls:
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
            log_rank_0(f"Auto-wrap policy: {layer_cls}")

    # Mixed precision policy
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # FSDP wrapping
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        # sync_module_states=True,
        # use_orig_params=True,
    )

    log_rank_0("FSDP1 wrapping completed!")
    return fsdp_model


def create_dummy_batch(num_sequences: int, sequence_length: int, vocab_size: int) -> dict:
    """Create dummy batch like in sequence_length_experiment.py."""
    total_tokens = num_sequences * sequence_length

    # Random input_ids and labels
    input_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.long)

    # Position_ids that reset for each sequence
    position_ids = []
    for _ in range(num_sequences):
        position_ids.extend(range(sequence_length))
    position_ids = torch.tensor(position_ids, dtype=torch.long)

    return {
        "input_ids": input_ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
    }


def test_forward_backward(model, optimizer, vocab_size: int):
    """Test single forward/backward pass."""
    log_rank_0("Testing forward/backward pass...")

    device = torch.cuda.current_device()

    # Create dummy batch
    batch = create_dummy_batch(num_sequences=4, sequence_length=512, vocab_size=vocab_size)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    try:
        # Forward pass
        output = model(**batch)
        loss = output.loss.float().sum()
        log_rank_0(f"Forward pass successful, loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        log_rank_0("Backward pass successful")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        log_rank_0("Optimizer step successful")

        # Memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        log_rank_0(f"Peak memory: {peak_memory:.2f}GB")

        return True

    except Exception as e:
        log_rank_0(f"Test failed: {e}")
        return False


def main():
    """Main function."""
    # Initialize distributed
    init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log_rank_0(f"Initialized: rank={rank}, world_size={world_size}")

    # Load model
    from transformers import AutoModelForCausalLM

    model_name = "/dev/shm/Qwen2.5-32B-Instruct"
    log_rank_0(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable()

    vocab_size = model.config.vocab_size
    log_rank_0(f"Model loaded, vocab_size: {vocab_size}")

    # Wrap with FSDP1
    fsdp_model = wrap_model_with_fsdp1(model)

    # Setup optimizer
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-5)

    # Test forward/backward
    success = test_forward_backward(fsdp_model, optimizer, vocab_size)

    if success:
        log_rank_0("✓ FSDP1 smoke test PASSED!")
    else:
        log_rank_0("✗ FSDP1 smoke test FAILED!")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
