"""
FSDP1 wrapper implementation - moved from main setup_model_for_training.py
This is kept for research/experimentation purposes.
"""

import functools

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from utils import log_rank_0


def get_module_class_from_name(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    modules_children = list(model.children())

    if model.__class__.__name__ == name:
        return model.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def wrap_fsdp1(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap `model` in PyTorch FSDP1 with full sharding and transformer auto-wrap policy under BF16.
    """
    # Enable gradient checkpointing for FSDP1 (HuggingFace style)
    log_rank_0("FSDP1: Enabling gradient checkpointing")
    model.gradient_checkpointing_enable()

    # Determine the block class to auto-wrap (first no-split module)
    if not model._no_split_modules:
        raise ValueError("model._no_split_modules is empty; cannot determine FSDP block class")
    block_name = next(iter(model._no_split_modules))
    block_cls = get_module_class_from_name(model, block_name)
    if block_cls is None:
        raise ValueError(f"Could not find module class named {block_name}")
    log_rank_0(f"FSDP1: Block class: {block_cls}")

    # Create auto-wrap policy using functools.partial
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})

    # Mixed-precision policy for BF16
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # FSDP1 wrapping
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
