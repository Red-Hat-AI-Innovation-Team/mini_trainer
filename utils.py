from datetime import timedelta
import importlib
import inspect
import logging
import os
from typing import Any

import torch
from torch.distributed import is_initialized, get_rank
import torch.distributed as dist
from rich.logging import RichHandler

def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"

def log_rank_0(msg, include_caller=False, rank=None, to_print=True):
    if rank is None:
        rank = get_rank() if is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logging.info(msg)

def setup_logger(level="DEBUG"):
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

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
    device = torch.device("cuda", dist.get_rank())
    t = torch.tensor([1]).to(device, torch.int32)
    
    # Here, every process group increments the counter
    # so the total amount should equal the world size.
    # all_reduce here is functionally equivalent to `dist.barrier`
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # We should see that all GPUs add the value up to 8
    assert t.item() == dist.get_world_size(), "❌ Error: distributed check failed"


def init_distributed_environment():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(minutes=180), device_id=device
    )
    # NOTE(osilkin): PyTorch wants us to avoid this API in favor of setting the device explicitly
    # through `init_process_group`, but without setting this, FSDP2 will shard the
    # entire model onto the first GPU. I haven't yet figured out a solution to this.
    torch.cuda.set_device(local_rank)
    check_distributed_is_synchronized()
    log_rank_0("✅ Torch distributed appears to be functioning correctly")

    torch.distributed.barrier()
