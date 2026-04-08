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
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
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

_DEFAULT_TRIGGER_PATH = "/dev/shm/mini_trainer_checkpoint_trigger"


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
            return trigger_tensor.item() > 0
        return local_triggered

    def cleanup(self):
        """Remove the trigger file if it exists."""
        self._trigger_path.unlink(missing_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_state: dict,
        checkpointer_state: dict,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
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
        """
        step = training_state["step"]
        save_dir = self._checkpoint_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Rank %d: saving full-state checkpoint to %s", self._rank, save_dir)

        # 1. Save model + optimizer state via DCP (sharded, all ranks participate)
        model_state = get_model_state_dict(model)
        optim_state = get_optimizer_state_dict(model, optimizer)
        dcp.save(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=str(save_dir / "distributed"),
        )

        # 2. Save per-rank RNG states
        self._save_rng_states(save_dir)

        # 3. Save global metadata (rank 0 only)
        sampler_epoch = getattr(data_loader.sampler, "_epoch", 0)
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

    def _save_rng_states(self, save_dir: Path):
        """Save per-rank RNG states for exact reproducibility."""
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
        optim_state = get_optimizer_state_dict(model, optimizer)
        dcp.load(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=dcp_dir,
        )
        set_model_state_dict(model, model_state)
        set_optimizer_state_dict(model, optimizer, optim_state)
