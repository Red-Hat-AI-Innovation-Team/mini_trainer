"""
Full-state on-demand checkpointing for training resumption.

Saves complete training state (model, optimizer, scheduler, RNG) via DCP
sharded saves when a termination signal is received. Each rank saves its
own shard — no gathering to rank 0.

Trigger mechanism: signal handler writes a file to /dev/shm (or configurable
path). Workers poll this file at batch boundaries and coordinate via
all_reduce(MAX) to ensure all ranks agree.
"""

import logging
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)

logger = logging.getLogger(__name__)

# Signals that indicate the training process should checkpoint and exit.
# Covers: Kubernetes (SIGTERM), SLURM (SIGUSR1/2), LSF (SIGXCPU, SIGQUIT),
# interactive (SIGINT), terminal disconnect (SIGHUP).
_CHECKPOINT_SIGNALS = (
    signal.SIGTERM,
    signal.SIGINT,
    signal.SIGUSR1,
    signal.SIGUSR2,
    signal.SIGXCPU,
    signal.SIGHUP,
    signal.SIGQUIT,
)

_DEFAULT_TRIGGER_PATH = f"/dev/shm/{os.environ.get('CHECKPOINT_TRIGGER_FILENAME', 'checkpoint_requested')}"


class FullStateCheckpointer:
    """Manages signal-driven full-state checkpointing for distributed training.

    On receiving a termination signal, writes a trigger file. At batch
    boundaries, workers poll this file and coordinate via all_reduce(MAX)
    to save a sharded DCP checkpoint before exiting.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        rank: int,
        world_size: int,
        trigger_path: str = _DEFAULT_TRIGGER_PATH,
    ):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._rank = rank
        self._world_size = world_size
        self._trigger_path = Path(trigger_path)
        self._original_handlers: dict[signal.Signals, signal._HANDLER] = {}

    def install_signal_handlers(self):
        """Register signal handlers that write the trigger file on receipt."""
        for sig in _CHECKPOINT_SIGNALS:
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                # Some signals may not be settable (e.g., in non-main thread)
                pass

    def _handle_signal(self, signum, frame):
        """Write trigger file atomically on signal receipt."""
        try:
            self._trigger_path.touch()
        except OSError:
            pass

    def should_save(self, device: torch.device) -> bool:
        """Check if a checkpoint save has been triggered.

        Polls the trigger file locally, then coordinates with all ranks
        via all_reduce(MAX) so all ranks agree.

        Args:
            device: The device to use for the all_reduce tensor.

        Returns:
            True if any rank detected the trigger.
        """
        local_triggered = self._trigger_path.exists()
        if dist.is_initialized():
            trigger_tensor = torch.tensor(int(local_triggered), dtype=torch.int32, device=device)
            dist.all_reduce(trigger_tensor, op=dist.ReduceOp.MAX)
            should_save = trigger_tensor.item() > 0
        else:
            should_save = local_triggered

        # Clear trigger file immediately to prevent stale triggers
        # across restarts or subsequent should_save() calls.
        if should_save:
            self._trigger_path.unlink(missing_ok=True)
        return should_save

    def cleanup(self):
        """Remove the trigger file and restore original signal handlers."""
        self._trigger_path.unlink(missing_ok=True)
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
        self._original_handlers.clear()

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_state: dict,
        checkpointer_state: dict,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        rng_snapshot: dict | None = None,
    ):
        """Save complete training state for exact resumption.

        Model and optimizer state are saved sharded via DCP (each rank saves
        its own shard). RNG states are saved per-rank. Training metadata is
        saved by rank 0 only.

        Args:
            model: The FSDP2-wrapped model.
            optimizer: The optimizer (with OSFT wrapping if applicable).
            lr_scheduler: The learning rate scheduler.
            training_state: Dict with step, epoch, total_samples_accumulated,
                total_tokens_processed, last_validation_loss.
            checkpointer_state: Dict with last_saved_samples,
                last_frequency_saved_samples, best_val_loss.
            data_loader: The training data loader (for sampler state).
            device: The device for distributed operations.
            rng_snapshot: Pre-captured RNG state from the start of the current
                training step. If None, captures the current RNG state (which
                may be mid-step and incorrect for replay).
        """
        step = training_state["step"]
        save_dir = self._checkpoint_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Rank %d: saving full-state checkpoint to %s", self._rank, save_dir)

        # 1. Save model state via DCP (sharded, all ranks participate)
        model_state = get_model_state_dict(model)
        dcp.save(
            {"model": model_state},
            checkpoint_id=str(save_dir / "distributed"),
        )

        # 2. Save optimizer state per-rank via torch.save.
        # We extract local shards from DTensors because:
        # - DCP requires the target state_dict to already have entries, but
        #   on resume the fresh optimizer has empty state.
        # - Plain torch.save of DTensors doesn't preserve sharding correctly.
        torch.save(
            self._extract_local_optim_state(optimizer),
            save_dir / f"optimizer_rank_{self._rank}.pt",
        )

        # 3. Save per-rank RNG states (use snapshot from step start if provided)
        self._save_rng_states(save_dir, rng_snapshot=rng_snapshot)

        # 3. Save model config for architecture consistency on resume
        # The model may have been modified (e.g., vocab resize) during the first
        # run, so we save the config to ensure resume loads the same architecture.
        inner = getattr(model, "module", model)
        if self._rank == 0 and hasattr(inner, "config"):
            inner.config.save_pretrained(save_dir)

        # 4. Save global metadata (rank 0 only)
        sampler_epoch = data_loader.sampler.epoch
        if self._rank == 0:
            self._save_metadata(
                save_dir=save_dir,
                training_state=training_state,
                checkpointer_state=checkpointer_state,
                lr_scheduler_state=lr_scheduler.state_dict(),
                sampler_epoch=sampler_epoch,
            )

        if dist.is_initialized():
            dist.barrier()
        logger.info("Rank %d: full-state checkpoint saved successfully", self._rank)

    def _save_metadata(
        self,
        save_dir: Path,
        training_state: dict,
        checkpointer_state: dict,
        lr_scheduler_state: dict,
        sampler_epoch: int,
    ):
        """Save training metadata (rank 0 only)."""
        metadata = {
            **training_state,
            "checkpointer_state": checkpointer_state,
            "lr_scheduler_state": lr_scheduler_state,
            "sampler_epoch": sampler_epoch,
        }
        torch.save(metadata, save_dir / "training_state.pt")

    def _save_rng_states(self, save_dir: Path, rng_snapshot: dict | None = None):
        """Save per-rank RNG states for exact reproducibility.

        If rng_snapshot is provided, saves that (captured at step start) instead
        of the current live RNG state. This is critical because the checkpoint
        may fire mid-accumulation, and resume needs to replay from step start.
        """
        if rng_snapshot is not None:
            torch.save(rng_snapshot, save_dir / f"rng_state_rank_{self._rank}.pt")
            return

        rng_state = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        # CUDA RNG state is only available if CUDA is initialized
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            rng_state["cuda_rng"] = torch.cuda.get_rng_state()
        torch.save(rng_state, save_dir / f"rng_state_rank_{self._rank}.pt")

    @staticmethod
    def load_metadata(checkpoint_dir: str | Path) -> dict:
        """Load training metadata from a checkpoint directory.

        Args:
            checkpoint_dir: Path to the checkpoint directory (e.g., step_10/).

        Returns:
            Dict containing training_state, checkpointer_state,
            lr_scheduler_state, sampler_epoch.
        """
        checkpoint_dir = Path(checkpoint_dir)
        meta_path = checkpoint_dir / "training_state.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"No training_state.pt found in {checkpoint_dir}")
        return torch.load(meta_path, weights_only=False)

    @staticmethod
    def load_rng_states(checkpoint_dir: str | Path, rank: int):
        """Restore per-rank RNG states from a checkpoint.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            rank: The rank whose RNG state to load.
        """
        checkpoint_dir = Path(checkpoint_dir)
        rng_path = checkpoint_dir / f"rng_state_rank_{rank}.pt"
        if not rng_path.exists():
            raise FileNotFoundError(f"No rng_state_rank_{rank}.pt found in {checkpoint_dir}")

        rng_state = torch.load(rng_path, weights_only=False)
        torch.set_rng_state(rng_state["torch_rng"])
        random.setstate(rng_state["python_rng"])
        np.random.set_state(rng_state["numpy_rng"])
        if "cuda_rng" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda_rng"])

    @staticmethod
    def _extract_local_optim_state(optimizer):
        """Extract optimizer state with DTensors converted to local CPU shards."""
        from torch.distributed._tensor import DTensor

        sd = optimizer.state_dict()
        local_sd = {"state": {}, "param_groups": sd["param_groups"]}
        for key, state in sd["state"].items():
            local_state = {}
            for skey, val in state.items():
                if isinstance(val, DTensor):
                    local_state[skey] = val._local_tensor.clone().cpu()
                elif isinstance(val, torch.Tensor):
                    local_state[skey] = val.clone().cpu()
                else:
                    local_state[skey] = val
            local_sd["state"][key] = local_state
        return local_sd

    @staticmethod
    def load_optimizer_state(
        checkpoint_dir: str | Path,
        optimizer: torch.optim.Optimizer,
        rank: int,
    ):
        """Load per-rank optimizer state, reconstructing DTensors from local shards.

        Requires the optimizer to already have state entries (do a dummy step first
        if the optimizer is fresh).

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            optimizer: The optimizer to load state into.
            rank: The rank whose optimizer state to load.
        """
        from torch.distributed._tensor import DTensor

        checkpoint_dir = Path(checkpoint_dir)
        optim_path = checkpoint_dir / f"optimizer_rank_{rank}.pt"
        loaded_sd = torch.load(optim_path, weights_only=False)

        # Write loaded state directly into optimizer.state, bypassing
        # load_state_dict which may deep-copy and alter DTensor values.
        # The optimizer.state uses parameter objects as keys; we iterate
        # param_groups to get the correct param-to-index mapping.
        param_list = [p for group in optimizer.param_groups for p in group["params"]]
        for idx_str, state in loaded_sd["state"].items():
            idx = int(idx_str) if isinstance(idx_str, str) else idx_str
            if idx >= len(param_list):
                continue
            param = param_list[idx]
            if param not in optimizer.state:
                optimizer.state[param] = {}
            for skey, val in state.items():
                cur = optimizer.state[param].get(skey)
                if cur is not None and isinstance(cur, DTensor):
                    # Copy loaded local shard into existing DTensor's storage
                    cur._local_tensor.copy_(val.to(cur._local_tensor.device))
                elif isinstance(val, torch.Tensor):
                    optimizer.state[param][skey] = val.to(param.device)
                else:
                    optimizer.state[param][skey] = val

    @staticmethod
    def load_distributed_state(
        checkpoint_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """Load sharded model and optimizer state from DCP checkpoint.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            model: The FSDP2-wrapped model (must have same structure as saved).
            optimizer: The optimizer (must match model parameters).
        """
        checkpoint_dir = Path(checkpoint_dir)
        dcp_dir = str(checkpoint_dir / "distributed")

        model_state = get_model_state_dict(model)
        optim_state = optimizer.state_dict()
        dcp.load(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=dcp_dir,
        )
        set_model_state_dict(model, model_state)
        optimizer.load_state_dict(optim_state)
