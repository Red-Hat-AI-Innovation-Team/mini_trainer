import importlib
import inspect
import logging
import os
import random
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from rich.logging import RichHandler
from torch.distributed import is_initialized
from transformers import AutoConfig
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING

_CONTROL_PROCESS_GROUP = None


def get_control_process_group():
    """
    Lazily create a CPU-friendly (Gloo) process group for control traffic.
    """
    global _CONTROL_PROCESS_GROUP
    if _CONTROL_PROCESS_GROUP is None:
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized before creating control group")
        ranks = list(range(dist.get_world_size()))
        _CONTROL_PROCESS_GROUP = dist.new_group(ranks=ranks, backend="gloo")
    return _CONTROL_PROCESS_GROUP


def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"


def log_rank_0(msg, include_caller=False, rank=None, to_print=True):
    if rank is None:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        rank = local_rank if is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logging.info(msg)


def setup_logger(level="DEBUG"):
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


def get_node_rank() -> int:
    # If torchrun was given --node_rank, this is usually exported:
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])

    # rank assignment is contiguous per node: rank = node_rank * nproc_per_node + local_rank
    # torchrun exports LOCAL_WORLD_SIZE == nproc_per_node
    return int(os.environ["RANK"]) // int(os.environ["LOCAL_WORLD_SIZE"])


def patch_target_module(
    to_patch: str,
    replace_with: Any,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)


def check_distributed_is_synchronized():
    """
    This function runs a simple check to verify that torch.distributed
    is functioning properly and all processes are synchronized.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    t = torch.tensor([1]).to(device, torch.int32)

    # Here, every process group increments the counter
    # so the total amount should equal the world size.
    # all_reduce here is functionally equivalent to `dist.barrier`
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # We should see that all GPUs add the value up to 8
    assert t.item() == dist.get_world_size(), "❌ Error: distributed check failed"


def check_distributed_is_evenly_configured():
    """
    DDP, FSDP1, and FSDP2 do not support uneven world-size configurations,
    and therefore neither do our distributed computing algorithms (e.g. distributed SVD init).
    PyTorch/torchrun should be enforcing this by default, but we double-check this here
    in case PyTorch ever changes their APIs or stops enforcing it.
    """
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # check that world_size is cleanly divisible by device count here:
    if world_size % local_world_size != 0:
        raise ValueError(
            f"world_size ({world_size}) is not cleanly divisible by local_world_size ({local_world_size}). Each node must have the same number of GPUs."
        )

    device = torch.device("cuda", local_rank)
    max_local_rank_seen = torch.tensor([local_rank], dtype=torch.int32, device=device)
    dist.all_reduce(max_local_rank_seen, op=dist.ReduceOp.MAX)
    if max_local_rank_seen[0] != local_world_size - 1:
        raise ValueError(
            f"max_local_rank_seen ({max_local_rank_seen[0]}) is not equal to local_world_size ({local_world_size}). Each node must have the same number of GPUs."
        )


def init_distributed_environment():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180), device_id=device)
    # NOTE(osilkin): PyTorch wants us to avoid this API in favor of setting the device explicitly
    # through `init_process_group`, but without setting this, FSDP2 will shard the
    # entire model onto the first GPU. I haven't yet figured out a solution to this.
    torch.cuda.set_device(local_rank)
    check_distributed_is_synchronized()
    check_distributed_is_evenly_configured()
    log_rank_0("✅ Torch distributed appears to be functioning correctly")

    torch.distributed.barrier()


def get_model_class_from_config(model_path):
    """Get the actual model class (not just the name) from a pretrained path.

    Note: vlm_utils.is_vlm_with_causal_lm() handles the broader VLM detection
    (deciding *whether* to extract).  This function resolves the model CLASS
    regardless of VLM status, falling back to text_config when needed.
    """
    # get the model class from config
    # TODO: make the `trust_remote_code` setting configurable somehow
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    config_class = config.__class__
    if config_class in mapping:
        resolved_cls = mapping[config_class]
        # Some models (e.g. Gemma 3) are dual-registered: the top-level config
        # maps to a ForConditionalGeneration VLM, not a text-only CausalLM.
        # In that case, prefer the text_config's CausalLM class instead.
        if "ForConditionalGeneration" in resolved_cls.__name__:
            text_config = getattr(config, "text_config", None)
            if text_config is not None and text_config.__class__ in mapping:
                return mapping[text_config.__class__]
        return resolved_cls

    # Fallback: for VLM models that wrap a CausalLM text backbone
    # (e.g. Mistral3Config wrapping Ministral3Config), check text_config
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config.__class__ in mapping:
        return mapping[text_config.__class__]

    # Fallback: for VLMs with no CausalLM class at all (e.g. Qwen3-VL-2B),
    # check the ImageTextToText mapping so they can be loaded directly.
    from transformers.models.auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    if config_class in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
        return MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[config_class]

    raise ValueError(f"Model class {config_class} not found in mapping {mapping}")


def destroy_distributed_environment():
    # wait for checkpoints to show up, once training is complete we tear it down
    dist.barrier()
    log_rank_0("Training complete 😀, tearing down distributed environment")
    dist.destroy_process_group()


def set_seed(seed: int):
    """
    This function sets the seed for the random number generators in the standard library,
    NumPy, and PyTorch.

    Args:
        seed: The seed to set.
    """
    # Reproducibility: align with HF Trainer seeding behavior
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
