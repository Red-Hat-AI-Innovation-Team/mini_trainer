"""Accelerator abstraction so training runs on CUDA or Intel XPU.

Everything device-specific (device type, torch device module, collective
backend) is resolved here so the rest of mini_trainer stays accelerator-neutral.

CUDA remains the default: XPU is selected only when an XPU is present and CUDA
is not, so behavior on CUDA and CPU-only hosts is unchanged.
"""

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

# Hugging Face hub flash-attention kernel with Intel XPU builds: the XPU
# counterpart of flash_attention_2 (the flash_attn package is CUDA/ROCm only).
XPU_FLASH_ATTN_KERNEL = "kernels-community/flash-attn2"


def get_device_type() -> str:
    """Return the accelerator type, ``"cuda"`` or ``"xpu"``.

    Set ``MINI_TRAINER_DEVICE`` to force a specific type.
    """
    forced = os.environ.get("MINI_TRAINER_DEVICE")
    if forced:
        return forced.lower()
    if not torch.cuda.is_available() and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda"


def get_device_module():
    """Return ``torch.cuda`` or ``torch.xpu`` for the active accelerator."""
    return getattr(torch, get_device_type())


def is_accelerator_available() -> bool:
    return get_device_module().is_available()


def get_device(local_rank: int | None = None) -> torch.device:
    """Return the accelerator ``torch.device`` for this rank (``LOCAL_RANK`` when not given)."""
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(get_device_type(), local_rank)


def get_dist_backend() -> str:
    """Return the collective backend matching the accelerator."""
    return {"cuda": "nccl", "xpu": "xccl"}[get_device_type()]


def set_device(local_rank: int) -> None:
    get_device_module().set_device(local_rank)


def empty_cache() -> None:
    get_device_module().empty_cache()


def manual_seed_all(seed: int) -> None:
    get_device_module().manual_seed_all(seed)


def memory_allocated(device: torch.device | None = None) -> int:
    return get_device_module().memory_allocated(device)


def max_memory_allocated() -> int:
    return get_device_module().max_memory_allocated()


def reset_peak_memory_stats() -> None:
    get_device_module().reset_peak_memory_stats()


def is_initialized() -> bool:
    module = get_device_module()
    return module.is_available() and module.is_initialized()


def get_rng_state() -> torch.Tensor:
    return get_device_module().get_rng_state()


def set_rng_state(state: torch.Tensor) -> None:
    get_device_module().set_rng_state(state)


@functools.cache
def xpu_flash_attn_implementation() -> str:
    """Attention implementation for padding-free (packed) training on XPU.

    Returns the hub flash-attention kernel when it can be loaded (needs the
    ``kernels`` package, see the ``xpu`` extra), else ``"sdpa"``. Both keep
    packed sequences isolated through ``position_ids``.
    """
    try:
        from transformers.integrations.hub_kernels import load_and_register_attn_kernel

        load_and_register_attn_kernel(XPU_FLASH_ATTN_KERNEL)
        return XPU_FLASH_ATTN_KERNEL
    except Exception as exc:
        logger.warning(
            "Flash attention kernel %s unavailable on XPU (%s); using SDPA instead. "
            "Install it with `pip install rhai-innovation-mini-trainer[xpu]`.",
            XPU_FLASH_ATTN_KERNEL,
            exc,
        )
        return "sdpa"
