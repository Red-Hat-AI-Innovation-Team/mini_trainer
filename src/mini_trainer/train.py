import logging
import time
import os
import sys
from pathlib import Path
import json
from typing import Annotated, Literal
from typer import Typer, Option

from mini_trainer.async_structured_logger import AsyncStructuredLogger
from mini_trainer import wandb_wrapper, mlflow_wrapper
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.distributed._tensor.api import (
    DTensor as _DTensor,
)  # works if DTensor is available

from mini_trainer.batch_metrics import BatchMetrics
from mini_trainer.sampler import get_data_loader
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import (
    init_distributed_environment,
    log_rank_0,
    setup_logger,
    get_node_rank,
    destroy_distributed_environment,
    set_seed,
)
from mini_trainer.training_types import TrainingMode, PretrainingConfig

# Suppress verbose HTTP request logs from httpx (used by huggingface_hub in transformers v5+)
logging.getLogger("httpx").setLevel(logging.WARNING)

SaveType = Literal["min_samples", "epoch", "final", "best_val_loss"]

app = Typer(
    pretty_exceptions_enable=False,  # disable rich exception formatting
)


def validate_training_state(
    model,
    optimizer,
    expected_param_dtype=torch.float32,
    expected_optimizer_dtype=torch.float32,
):
    """
    Validates that the model parameters and optimizer state are in their expected dtypes.

    Args:
        model: The model to validate
        optimizer: The optimizer to validate
        expected_param_dtype: Expected dtype for model parameters and gradients
        expected_optimizer_dtype: Expected dtype for optimizer state (usually float32 for numerical stability)
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != expected_param_dtype:
            raise ValueError(
                f"Parameter {name} is not in {expected_param_dtype}, got {param.dtype}"
            )
        if param.grad is not None and param.grad.dtype != expected_param_dtype:
            raise ValueError(
                f"Gradient {name} is not in {expected_param_dtype}, got {param.grad.dtype}"
            )

    # Check optimizer state tensors - only for trainable parameters
    for p_obj, state in optimizer.state.items():
        # Skip optimizer states for frozen parameters (e.g., GPT-OSS router params)
        if hasattr(p_obj, "requires_grad") and not p_obj.requires_grad:
            continue

        for k, v in state.items():
            # Skip non-tensor entries and special scalar fields
            if not (torch.is_tensor(v) or isinstance(v, _DTensor)):
                continue

            # Skip specific optimizer state fields that should not match parameter dtype
            # These include step counters and other scalar-like fields
            if k in ["step"]:
                continue

            # Only validate gradient momentum tensors (exp_avg, exp_avg_sq, etc.)
            # These should match the parameter dtype in mixed precision training
            v_dtype = v.dtype
            if v_dtype != expected_optimizer_dtype:
                raise ValueError(
                    f"Optimizer state {k} is not in {expected_optimizer_dtype} (got {v_dtype})"
                )


def take_gradient_step(model, optimizer, lr_scheduler, expected_dtype=torch.float32):
    """Scales gradients, applies clipping, and takes an optimization step."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    validate_training_state(
        model,
        optimizer,
        expected_param_dtype=expected_dtype,
        expected_optimizer_dtype=expected_dtype,
    )
    optimizer.zero_grad()
    return grad_norm


def save_model(
    fsdp_model,
    samples_seen: int,
    output_dir: str,
    model_name_or_path: str,
    suffix: str | None = None,
):
    """
    Save the given FSDP Model as a checkpoint in HF Format.

    Args:
        fsdp_model (str): The model to save.
        samples_seen (int): The number of samples seen so far.
        output_dir (str): The directory to save the model.
        model_name_or_path (str): The model name or path.
        suffix (str | None): Optional suffix to add to the checkpoint directory name.
    """
    from huggingface_hub import split_torch_state_dict_into_shards
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    from mini_trainer.gpt_oss_utils import (
        is_gpt_oss_model,
        convert_dequantized_to_quantized_format_correct,
    )

    # Only on rank 0
    suffix_text = f" ({suffix})" if suffix else ""
    log_rank_0(f"Saving model at {samples_seen} samples{suffix_text}")
    start = time.time()
    global_rank = torch.distributed.get_rank()

    # Add suffix to directory name if provided
    dir_name = f"samples_{samples_seen}"
    if suffix:
        dir_name += f"_{suffix}"
    save_directory = Path(output_dir) / "hf_format" / dir_name
    os.makedirs(save_directory, exist_ok=True)

    # NOTE(osilkin):
    # Here, we gather the model's state-dict and offload it onto the CPU
    # The downside with this approach is that it requires recomputing
    # each OSFT parameter on the CPU.
    # This can be optimized by modifying the `prepare_state_dict_for_save` function so that it
    # processes weights on the GPU device in batches before de-allocating the memory being consumed
    # Users may also face issues here if they lack the CPU memory required to store the original
    # FP32 state dict on CPU.
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    state_dict = get_model_state_dict(
        fsdp_model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=False,
        ),
    )
    inner = getattr(fsdp_model, "module", fsdp_model)
    # save in whatever data type is stored on the model config
    # by now the `torch_dtype` attribute has been set to some value
    save_dtype = inner.config.torch_dtype

    # NOTE(osilkin): This save function could be further optimized for quicker checkpoints:
    #
    # FSDP2 provides a distributed checkpoint API, which allows all shards to
    # save their respective format, which can be post-processed afterwards
    # to recover the model and optionally the optimizer states.
    #
    # However; switching to this format would require:
    # 1.) Converting checkpoints into HF format after training completes
    # 2.) All nodes having access to the same write location, which is also synchronized for us
    #     to actually export the checkpoints properly.

    # Unfortunately this process takes quite a bit on the CPU. (~5 mins)
    if global_rank == 0:
        # only the main global process saves
        if hasattr(inner, "prepare_state_dict_for_save"):
            state_dict = inner.prepare_state_dict_for_save(state_dict)

    # This process takes a while, so worker nodes should print when this is happening
    if get_node_rank() != 0 and hasattr(inner, "prepare_state_dict_for_save"):
        log_rank_0(
            "Model checkpoint is being prepared on the main process of the master node. Please wait..."
        )

    torch.distributed.barrier()

    # Check if this is a GPT-OSS model that needs format conversion
    is_gpt_oss = is_gpt_oss_model(inner.config)

    if global_rank == 0:
        # Model format conversion (GPT-OSS vs standard)
        if is_gpt_oss:
            log_rank_0(
                "🔧 Converting GPT-OSS parameters to quantized format for compatibility"
            )
            # Convert state dict on GPU, then move to CPU
            state_dict = convert_dequantized_to_quantized_format_correct(state_dict)
        else:
            # Once we have all of our parameters, we need to ensure they're stored in BF16
            # so checkpoints aren't terrible heavy. We have to do this _after_ `prepare_state_dict_for_save`
            # has been called so we don't lose fidelity.
            notified_about_dtype = False
            cpu_device = torch.device("cpu")
            # Standard conversion to bf16 and CPU
            for k, v in state_dict.items():
                if v.dtype != save_dtype:
                    if not notified_about_dtype:
                        log_rank_0(
                            f"⚠️  Warning: Found tensor {k} with dtype {v.dtype}, casting to {save_dtype}"
                        )
                        notified_about_dtype = True
                    state_dict[k] = v.to(dtype=save_dtype, device=cpu_device)

        # All saving operations
        pattern = "model{suffix}.safetensors"
        index_name = "model.safetensors.index.json"

        # Shard splitting
        split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=pattern,
            max_shard_size="5GB",
        )
        # Save shards
        for filename, tensors in split.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            path = os.path.join(save_directory, filename)
            save_file(shard, path)

        # Save index if sharded
        if split.is_sharded:
            index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with open(os.path.join(save_directory, index_name), "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
        # Save config and tokenizer
        if is_gpt_oss:
            # For GPT-OSS models, add quantization config before saving (single write)
            config_dict = inner.config.to_dict()
            if "quantization_config" not in config_dict:
                config_dict["quantization_config"] = {
                    "modules_to_not_convert": [
                        "model.layers.*.self_attn",
                        "model.layers.*.mlp.router",
                        "model.embed_tokens",
                        "lm_head",
                    ],
                    "quant_method": "mxfp4",
                }
            # Save the modified config
            with open(os.path.join(save_directory, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            # Standard config save for non-GPT-OSS models
            inner.config.to_json_file(os.path.join(save_directory, "config.json"))

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(save_directory)

    if get_node_rank() != 0:
        log_rank_0(
            "Model checkpoint is being saved on the main process of the master node. Please wait..."
        )

    log_rank_0("")
    torch.distributed.barrier()
    log_rank_0(
        f"✅ Saved model at {samples_seen} samples{suffix_text} in {time.time() - start:.2f} seconds"
    )


def compute_validation_loss(model, val_data_loader, device):
    """Compute validation loss on the validation dataset with tqdm-styled progress bar.

    Args:
        model: The model to evaluate
        val_data_loader: Validation data loader
        device: Device to run evaluation on

    Returns:
        dict: Dictionary containing validation metrics
    """
    if val_data_loader is None:
        return {}

    log_rank_0("Computing validation loss...")
    model.eval()
    val_batch_totals = BatchMetrics()
    total_val_batches = 0
    total_num_tokens = 0
    total_overall_loss = 0.0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Get total number of batches for progress bar
    total_batches = len(val_data_loader)

    # Create tqdm with custom format matching Rich style
    val_bar_format = (
        "\033[1;35mValidation:\033[0m "
        "{bar} "
        "\033[33m{percentage:3.0f}%\033[0m │ "
        "\033[37m{n}/{total}\033[0m"
    )
    val_pbar = tqdm(
        total=total_batches,
        bar_format=val_bar_format,
        ncols=None,
        leave=False,
        position=0,
        file=sys.stdout,
        ascii="━╺─",  # custom characters matching Rich style
        disable=True,  # disable auto-display, we'll manually format
    )

    with torch.no_grad():
        val_data_loader_it = iter(val_data_loader)

        # For simplicity, we pack the batches needed for computing validation
        # loss in the same way as happens in training. As a result, it will
        # be collated in the same way: Each minibatch consists of at most `batch_size`
        # samples, which are then split into (num_gpus * grad_accum) pieces.
        for val_batch_idx, batch in enumerate(val_data_loader_it, 1):
            val_batch_totals.reset_batch()

            # Loss is accumulated WRT to the global number of samples; i.e.
            # all of the loss-counted tokens in the entire validation set.
            # As a result, grad accum is ignored, and only the total loss + total num tokens
            # is used to calculate the actual loss.
            for _, mb in enumerate(batch):
                mb_num_loss_counted_tokens = mb["num_loss_counted_tokens"]
                mb_num_samples = mb["num_samples"]

                # Send inputs to device
                model_inputs = {
                    "input_ids": mb["input_ids"].to(device),
                    "labels": mb["labels"].to(device),
                }

                # add optional fields onto `model_inputs` object
                pos_ids = mb.get("position_ids")
                if pos_ids is not None:
                    # position_ids gets used for padding-free training
                    model_inputs["position_ids"] = pos_ids.to(device)
                attn_mask = mb.get("attention_mask")
                if attn_mask is not None:
                    # attention_mask gets used for padded training
                    model_inputs["attention_mask"] = attn_mask.to(device)

                # Forward pass
                output = model(**model_inputs)
                loss = output.loss.float().sum()
                loss_metrics = loss.detach().item()

                # Clear cache after each minibatch to prevent OOM
                torch.cuda.empty_cache()

                val_batch_totals.accumulate_minibatch_metrics(
                    num_loss_counted_tokens=mb_num_loss_counted_tokens,
                    num_total_tokens=mb["input_ids"].numel(),
                    num_samples=mb_num_samples,
                    loss=loss_metrics,
                    loss_backward=0.0,  # No backward pass for validation
                    time_per_minibatch=0.0,  # Not tracking time for validation
                )

            # Reduce metrics across all processes
            torch.distributed.barrier()
            val_batch_totals.reduce_batch_metrics(device)
            total_val_batches += 1
            total_overall_loss += val_batch_totals.totals["loss"]

            # ensure there was an item in the batch
            assert len(batch) > 0, "validation batch was empty"
            total_num_tokens += batch[0]["batch_num_loss_counted_tokens"]

            # Print tqdm-styled validation progress
            if is_main_process:
                current_loss = (
                    total_overall_loss / total_num_tokens
                    if total_num_tokens > 0
                    else 0.0
                )
                val_pbar.n = val_batch_idx

                # Manually format the complete progress line with loss metric using format_meter
                bar_str = val_pbar.format_meter(
                    n=val_batch_idx,
                    total=total_batches,
                    elapsed=0,
                    ncols=None,
                    bar_format=val_bar_format,
                    ascii="━╺─",
                )

                # Add the loss metric
                metrics_str = (
                    f" │ \033[32mloss:\033[0m \033[37m{current_loss:.4f}\033[0m"
                )

                # Print the complete line
                print(bar_str + metrics_str, file=sys.stdout, flush=True)

            dist.barrier()

    # Calculate average validation metrics and synchronize across all processes
    vbm = val_batch_totals.totals
    if total_val_batches > 0 and vbm["num_loss_counted_tokens"] > 0:
        avg_val_loss = total_overall_loss / total_num_tokens
        val_metrics = {
            "val_loss": avg_val_loss,
            "val_num_samples": vbm["num_samples"],
            "val_num_loss_counted_tokens": vbm["num_loss_counted_tokens"],
            "val_num_batches": total_val_batches,
        }
        log_rank_0(f"Validation loss: {avg_val_loss:.6f}")
    else:
        val_metrics = {}
        log_rank_0("No validation data processed")

    model.train()  # Set back to training mode
    return val_metrics


def reached_stop_condition(
    training_mode: TrainingMode,
    current_epoch: int,
    current_step: int,
    tokens_seen: int,
    max_epochs: int = 0,
    max_steps: int = 0,
    max_tokens: int = 0,
) -> bool:
    """
    Convenience function which determines whether or not training has reached
    a stopping condition based on the training mode.

    Args:
        training_mode: The training mode (EPOCH, STEP, TOKEN, or INFINITE)
        current_epoch: Current epoch number
        current_step: Current step number
        tokens_seen: Total number of loss-counted tokens processed so far
        max_epochs: Maximum epochs (for EPOCH mode)
        max_steps: Maximum steps (for STEP mode)
        max_tokens: Maximum tokens (for TOKEN mode)

    Returns:
        bool: True if stopping condition is reached, False otherwise
    """
    match training_mode:
        case TrainingMode.EPOCH:
            return max_epochs > 0 and current_epoch >= max_epochs
        case TrainingMode.STEP:
            return max_steps > 0 and current_step >= max_steps
        case TrainingMode.TOKEN:
            return max_tokens > 0 and tokens_seen >= max_tokens
        case TrainingMode.INFINITE:
            return False  # Never stop for infinite mode
        case _:
            raise ValueError(f"Unknown training mode: {training_mode}")


def parse_dtype(dtype_input: str | None) -> torch.dtype | None:
    """Convert string dtype to torch dtype.

    Args:
        dtype_input: String representation of dtype, torch.dtype object, or None

    Returns:
        torch.dtype object or None

    Raises:
        ValueError: If dtype_input is an unsupported string
        TypeError: If dtype_input is not str, torch.dtype, or None
    """
    if dtype_input is None:
        return None
    if isinstance(dtype_input, str):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype_input not in dtype_map:
            raise ValueError(
                f"Unsupported dtype string: '{dtype_input}'. Supported dtypes: {list(dtype_map.keys())}"
            )
        return dtype_map[dtype_input]
    elif hasattr(dtype_input, "dtype") or str(type(dtype_input)).startswith(
        "<class 'torch."
    ):
        # Already a torch dtype
        return dtype_input
    else:
        raise TypeError(
            f"Invalid dtype type: {type(dtype_input)}. Expected str, None, or torch.dtype, got {dtype_input}"
        )


def validate_training_mode(
    training_mode: TrainingMode,
    max_epochs: int,
    max_steps: int,
    max_tokens: int,
) -> TrainingMode:
    """
    Validates that the given training mode is valid, and ensures that
    the provided options are being used as expected.

    When the training mode was provided as a string, this function
    returns the corresponding
    """
    # # Convert string training mode to TrainingMode enum if needed
    # if isinstance(training_mode, str):
    #     try:
    #         training_mode = TrainingMode(training_mode.lower())
    #     except ValueError:
    #         valid_modes = [mode.value for mode in TrainingMode]
    #         raise ValueError(f"Invalid training mode: '{training_mode}'. Valid modes are: {valid_modes}")

    # Validate training mode and corresponding parameters
    if training_mode == TrainingMode.EPOCH and max_epochs <= 0:
        raise ValueError("EPOCH training mode requires max_epochs > 0")
    elif training_mode == TrainingMode.STEP and max_steps <= 0:
        raise ValueError("STEP training mode requires max_steps > 0")
    elif training_mode == TrainingMode.TOKEN and max_tokens <= 0:
        raise ValueError("TOKEN training mode requires max_tokens > 0")


class Checkpointer:
    """
    A stateful checkpointer that manages when to save model checkpoints.

    This class consolidates the logic for deciding if a checkpoint should be saved
    and prevents duplicate checkpointing, which is particularly important for OSFT
    since checkpointing is an expensive operation.

    The checkpointer supports multiple save types:
    - min_samples: Save every N samples
    - epoch: Save at the end of each epoch
    - final: Save at the end of training
    - best_val_loss: Save when validation loss improves
    """

    def __init__(
        self,
        min_samples_per_checkpoint: int | None = None,
        save_best_val_loss: bool = False,
        val_loss_improvement_threshold: float = 0.0,
        checkpoint_at_epoch: bool = False,
        checkpoint_at_final: bool = False,
    ):
        """
        Initialize the checkpointer.

        Args:
            min_samples_per_checkpoint: Minimum samples between frequency-based saves
            save_best_val_loss: Whether to save on validation loss improvement
            val_loss_improvement_threshold: Minimum improvement needed to save (default: any improvement)
            checkpoint_at_epoch: Whether epoch-based checkpointing is enabled
            checkpoint_at_final: Whether we should save at the end of training
        """
        self.min_samples_per_checkpoint = min_samples_per_checkpoint
        self.save_best_val_loss = save_best_val_loss
        self.val_loss_improvement_threshold = val_loss_improvement_threshold

        # NOTE(osilkin): It feels like maybe a mistake to place the checkpointing settings here,
        # but it cleans up the training loop 🤷
        self.checkpoint_at_epoch = checkpoint_at_epoch
        self.checkpoint_at_final = checkpoint_at_final

        # State tracking
        self.last_saved_samples = 0
        self.last_frequency_saved_samples = 0
        self.best_val_loss: float | None = None

    def should_save_checkpoint(
        self,
        save_type: SaveType,
        accumulated_samples: int,
        end_of_epoch: bool = False,
        end_of_training: bool = False,
        val_loss: float | None = None,
    ) -> bool:
        """
        Determine if a checkpoint should be saved based on the save type and current state.

        Args:
            save_type: The type of checkpoint to consider
            accumulated_samples: Total samples processed so far
            end_of_epoch: Whether we're at the end of an epoch
            end_of_training: Whether training is ending
            val_loss: Current validation loss (if available)

        Returns:
            True if a checkpoint should be saved, False otherwise
        """
        match save_type:
            case "min_samples":
                return (
                    self.min_samples_per_checkpoint is not None
                    and accumulated_samples
                    >= self.last_frequency_saved_samples
                    + self.min_samples_per_checkpoint
                )

            case "epoch":
                # Have we processed any new information since the last checkpoint?
                if not self.checkpoint_at_epoch:
                    return False

                if not end_of_epoch:
                    return False

                return accumulated_samples > self.last_saved_samples

            case "final":
                if not self.checkpoint_at_final:
                    return False

                if not end_of_training:
                    return False
                return accumulated_samples > self.last_saved_samples

            case "best_val_loss":
                if not self.save_best_val_loss or val_loss is None:
                    return False

                # First validation loss - save it
                if self.best_val_loss is None:
                    return True

                # Check if validation loss improved enough
                improvement = self.best_val_loss - val_loss
                return improvement > self.val_loss_improvement_threshold

            case _:
                raise ValueError(f"Unknown save type: {save_type}")

    def record_save(
        self,
        save_type: SaveType,
        accumulated_samples: int,
        val_loss: float | None = None,
    ):
        """
        Record that a checkpoint was saved and update internal state.

        Args:
            save_type: The type of checkpoint that was saved
            accumulated_samples: Total samples processed when the save occurred
            val_loss: Validation loss at time of save (if applicable)
        """
        # Always update the general save tracker
        self.last_saved_samples = accumulated_samples

        # Update type-specific state
        if save_type == "min_samples":
            self.last_frequency_saved_samples = accumulated_samples
        elif save_type == "best_val_loss" and val_loss is not None:
            self.best_val_loss = val_loss
            log_rank_0(f"New best validation loss: {val_loss:.6f}")


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str,
    min_samples_per_checkpoint: int | None,
    model_name_or_path: str,
    training_mode: TrainingMode = TrainingMode.EPOCH,
    max_epochs: int = 1,
    max_steps: int = 0,
    max_tokens: int = 0,
    checkpoint_at_epoch: bool = False,
    save_final_checkpoint: bool = False,
    train_dtype: torch.dtype = torch.float32,
    save_best_val_loss: bool = False,
    val_loss_improvement_threshold: float = 0.0,
    use_wandb: bool = False,
    use_mlflow: bool = False,
    val_data_loader: torch.utils.data.DataLoader | None = None,
    validation_frequency: int | None = None,
):
    """
    Runs the model training loop.

    Runs the model training loop with FSDP (Fully Sharded Data Parallel) support.

    This function handles the complete training process including gradient accumulation,
    checkpointing, logging, and distributed training coordination. It supports four
    different training modes: epoch-based, step-based, token-based, and infinite.

    Args:
        model (torch.nn.Module): The model to train, typically wrapped with FSDP.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        output_dir (str): Directory path where checkpoints and logs will be saved.
        min_samples_per_checkpoint (int | None): Minimum number of samples between checkpoints. If None, sample-based checkpointing is disabled.
        model_name_or_path (str): Path or identifier of the base model for tokenizer loading.
        training_mode (Union[TrainingMode, str], optional): Training mode - EPOCH, STEP, TOKEN, or INFINITE. Can be either a TrainingMode enum or string value. Defaults to INFINITE.
        max_epochs (int, optional): Maximum number of epochs (for EPOCH mode). Defaults to 0.
        max_steps (int, optional): Maximum number of steps (for STEP mode). Defaults to 0.
        max_tokens (int, optional): Maximum number of loss-counted tokens (for TOKEN mode). Defaults to 0.
        checkpoint_at_epoch (bool, optional): Whether to save checkpoints at epoch end. Defaults to False.
        save_final_checkpoint (bool, optional): Whether to save a final checkpoint at training end. Defaults to False.
        train_dtype (torch.dtype, optional): Dtype for training computations. Defaults to torch.float32.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.
        save_best_val_loss (bool, optional): Whether to save checkpoints when validation loss improves. Defaults to False.
        val_loss_improvement_threshold (float, optional): Minimum validation loss improvement required to trigger a save. Defaults to 0.0 (any improvement).
        val_data_loader (torch.utils.data.DataLoader | None, optional): Validation data loader. If provided, validation loss will be computed. Defaults to None.
        validation_frequency (int | None, optional): Frequency of validation evaluation in steps. Required when val_data_loader is provided. Defaults to None.

    Note:
        The training_mode can be provided as either a TrainingMode enum value or a string:
        - "epoch" or TrainingMode.EPOCH: requires max_epochs > 0
        - "step" or TrainingMode.STEP: requires max_steps > 0
        - "token" or TrainingMode.TOKEN: requires max_tokens > 0
        - "infinite" or TrainingMode.INFINITE: runs indefinitely until manually stopped

    Raises:
        RuntimeError: If distributed training is not properly initialized.
        ValueError: If training mode requirements are not met.
    """
    log_rank_0(f"Training model: {model_name_or_path}")

    # just ensure that the way we are being prompted to train is correct
    validate_training_mode(training_mode, max_epochs, max_steps, max_tokens)

    # set model into training mode
    model.train()

    # control args
    world_size = int(os.environ["WORLD_SIZE"])
    is_local_main_process = int(os.getenv("LOCAL_RANK", 0)) == 0
    metric_logger = AsyncStructuredLogger(
        output_dir + f"/training_metrics_{get_node_rank()}.jsonl",
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
    )

    # initialize variables
    batch_totals = BatchMetrics()
    step = 0
    total_samples_accumulated = 0
    total_tokens_processed = 0  # Track total loss-counted tokens for TOKEN mode

    # Initialize the checkpointer to manage saving logic
    checkpointer = Checkpointer(
        min_samples_per_checkpoint=min_samples_per_checkpoint,
        save_best_val_loss=save_best_val_loss,
        val_loss_improvement_threshold=val_loss_improvement_threshold,
        checkpoint_at_epoch=checkpoint_at_epoch,
        checkpoint_at_final=save_final_checkpoint,
    )

    device = next(model.parameters()).device
    epoch = 0
    last_validation_loss = None  # Track the most recent validation loss

    # main training loop
    while not reached_stop_condition(
        training_mode=training_mode,
        current_epoch=epoch,
        current_step=step,
        tokens_seen=total_tokens_processed,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
    ):
        # set the current epoch
        data_loader.sampler.set_epoch(epoch)
        data_loader_it = iter(data_loader)

        for batch in data_loader_it:
            batch_start_time = time.time()
            batch_totals.reset_batch()
            torch.cuda.reset_peak_memory_stats()
            for grad_accum, mb in enumerate(batch):
                mb_start_time = time.time()
                mb_num_loss_counted_tokens = mb["num_loss_counted_tokens"]
                mb_num_samples = mb["num_samples"]
                batch_num_loss_counted_tokens = mb["batch_num_loss_counted_tokens"]

                # be explicit about what gets sent to the device
                model_inputs = {
                    "input_ids": mb["input_ids"].to(device),
                    "labels": mb["labels"].to(device),
                }
                # position_ids is for padding-free training, attention_mask is for padded training
                if (pos_ids := mb.get("position_ids")) is not None:
                    model_inputs["position_ids"] = pos_ids.to(device)
                if (attn_mask := mb.get("attention_mask")) is not None:
                    model_inputs["attention_mask"] = attn_mask.to(device)

                output = model(**model_inputs)

                # GPT-OSS: add auxiliary loss if present, otherwise use standard loss
                loss = output.loss.float().sum()
                if hasattr(output, "aux_loss") and output.aux_loss is not None:
                    # MoE-style models will have an aux loss which we want to compute
                    loss += output.aux_loss.float().sum()

                # Ensure scalar loss even if model returns per-token loss
                loss = (loss / batch_num_loss_counted_tokens) * world_size
                loss_metrics = loss.detach().cpu().item()
                loss.backward()
                torch.cuda.empty_cache()

                batch_totals.accumulate_minibatch_metrics(
                    num_loss_counted_tokens=mb_num_loss_counted_tokens,
                    num_total_tokens=mb["input_ids"].shape[1],
                    num_samples=mb_num_samples,
                    loss=loss_metrics,
                    # since FSDP2 automatically averages gradients by the world-size,
                    # each rank's gradient contributes 1/8 to the backward
                    loss_backward=loss.detach().item() / world_size,
                    time_per_minibatch=time.time() - mb_start_time,
                )
            step += 1
            # sum the metrics from all processes
            batch_totals.reduce_batch_metrics(device)

            # use accumulated metrics to take a gradient step and logging
            bm = batch_totals.totals
            total_samples_accumulated += bm["num_samples"]
            total_tokens_processed += (
                batch_num_loss_counted_tokens  # Track tokens for TOKEN mode
            )

            # capture the LR that we'll use when taking a training step
            current_lr = lr_scheduler.get_last_lr()[0]
            grad_norm = take_gradient_step(
                model, optimizer, lr_scheduler, expected_dtype=train_dtype
            )

            batch_time = time.time() - batch_start_time

            # Since the loss accounts for FSDP2's world_size averaging, we need to adjust this back to
            # its original scale after adding up the losses across each proc
            logged_loss = bm["loss"] / world_size
            batch_metrics = {
                "step": step,
                "epoch": epoch,
                "steps_per_epoch": len(data_loader),
                "lr": current_lr,
                "grad_norm": grad_norm.item(),
                "loss": logged_loss,
                "avg_loss_backward": bm["loss_backward"] / (grad_accum + 1),
                "num_samples": bm["num_samples"],
                "num_loss_counted_tokens": bm["num_loss_counted_tokens"],
                "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
                "num_total_tokens": bm["num_total_tokens"],
                "grad_accum": grad_accum + 1,
                "avg_time_per_minibatch": bm["time_per_minibatch"]
                / (grad_accum + 1)
                / world_size,
                "time_per_batch": batch_time,
                "tokens_per_second": bm["num_total_tokens"] / batch_time,
                "total_samples_accumulated": total_samples_accumulated,
                "total_tokens_accumulated": total_tokens_processed,
                "samples_per_second": bm["num_samples"] / batch_time
                if batch_time > 0
                else 0.0,
                "peak_memory_usage_GB": float(torch.cuda.max_memory_allocated() / 1e9),
                "val_loss": last_validation_loss,
            }
            # Add validation metrics if it's time to validate
            if (
                val_data_loader is not None
                and validation_frequency is not None
                and step % validation_frequency == 0
            ):
                val_metrics = compute_validation_loss(model, val_data_loader, device)
                if val_metrics and "val_loss" in val_metrics:
                    last_validation_loss = val_metrics["val_loss"]
                    print(f"Validation loss: {last_validation_loss}")
                batch_metrics.update(val_metrics)

            # Log metrics (progress info is printed by the logger)
            if is_local_main_process:
                metric_logger.log_sync(batch_metrics)

            dist.barrier()

            # sample-based saving, keep in the inner loop
            if checkpointer.should_save_checkpoint(
                save_type="min_samples", accumulated_samples=total_samples_accumulated
            ):
                save_model(
                    model, total_samples_accumulated, output_dir, model_name_or_path
                )
                checkpointer.record_save("min_samples", total_samples_accumulated)

            # Check for best validation loss saving after validation runs
            if checkpointer.should_save_checkpoint(
                save_type="best_val_loss",
                accumulated_samples=total_samples_accumulated,
                val_loss=last_validation_loss,
            ):
                save_model(
                    model,
                    total_samples_accumulated,
                    output_dir,
                    model_name_or_path,
                    suffix="best_val_loss",
                )
                checkpointer.record_save(
                    "best_val_loss", total_samples_accumulated, last_validation_loss
                )

            torch.distributed.barrier()

            # Check stopping condition after each step (for STEP and TOKEN modes)
            if reached_stop_condition(
                training_mode=training_mode,
                current_epoch=epoch,
                current_step=step,
                tokens_seen=total_tokens_processed,
                max_epochs=max_epochs,
                max_steps=max_steps,
                max_tokens=max_tokens,
            ):
                break

        # Increment epoch counter after completing an epoch
        epoch += 1

        # save at the current number of samples seen
        # should save at the end of each epoch
        if checkpointer.should_save_checkpoint(
            save_type="epoch",
            accumulated_samples=total_samples_accumulated,
            end_of_epoch=True,
        ):
            save_model(model, total_samples_accumulated, output_dir, model_name_or_path)
            checkpointer.record_save("epoch", total_samples_accumulated)

    torch.distributed.barrier()
    # save one last time if we haven't yet
    if checkpointer.should_save_checkpoint(
        save_type="final",
        accumulated_samples=total_samples_accumulated,
        end_of_training=True,
    ):
        save_model(model, total_samples_accumulated, output_dir, model_name_or_path)
        checkpointer.record_save("final", total_samples_accumulated)


def calculate_num_training_steps(
    training_mode: TrainingMode,
    data_loader,
    max_epochs: int = 0,
    max_steps: int = 0,
    max_tokens: int = 0,
) -> int | None:
    """
    Calculate the number of training steps based on the training mode.

    Args:
        training_mode: The training mode (EPOCH, STEP, TOKEN, or INFINITE)
        data_loader: The data loader to get dataset statistics from
        max_epochs: Maximum epochs for EPOCH mode
        max_steps: Maximum steps for STEP mode
        max_tokens: Maximum tokens for TOKEN mode

    Returns:
        Number of training steps, or None for INFINITE mode or when it can't be calculated
    """

    if training_mode == TrainingMode.INFINITE:
        log_rank_0("INFINITE training mode: num_training_steps is None")
        return None

    # The most straightforward case
    elif training_mode == TrainingMode.STEP:
        log_rank_0(f"STEP training mode: num_training_steps = {max_steps}")
        return max_steps

    elif training_mode == TrainingMode.EPOCH:
        # Count the number of batches in one epoch
        num_training_steps = len(data_loader) * max_epochs
        log_rank_0(
            f"EPOCH training mode: {len(data_loader)} batches/epoch * {max_epochs} epochs = {num_training_steps} steps"
        )
        return num_training_steps

    elif training_mode == TrainingMode.TOKEN:
        # Calculate average tokens per batch
        log_rank_0("Calculating average tokens per batch...")
        total_loss_tokens = sum(
            mb[0]["batch_num_loss_counted_tokens"] for mb in data_loader
        )
        avg_tokens_per_batch = total_loss_tokens / len(data_loader)
        num_training_steps = int(max_tokens / avg_tokens_per_batch)  # approximate value
        log_rank_0(
            f"TOKEN training mode: {max_tokens} tokens / {avg_tokens_per_batch:.1f} avg tokens/batch = {num_training_steps} steps"
        )
        return num_training_steps

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


@app.command()
def main(
    # the '...' is a way of defining required options/arguments without breaking Python's
    # positional vs keyword argument rules
    model_name_or_path: Annotated[str, Option(help="Model name or path")] = ...,
    data_path: Annotated[
        str, Option(help="Path to the training data JSONL file")
    ] = ...,
    batch_size: Annotated[
        int, Option(help="Initial batch size before dynamic splitting")
    ] = ...,
    max_tokens_per_gpu: Annotated[
        int, Option(help="Maximum tokens per GPU per minibatch")
    ] = ...,
    learning_rate: Annotated[float, Option(help="Peak learning rate")] = ...,
    num_warmup_steps: Annotated[
        int, Option(help="Number of warmup steps for the LR scheduler")
    ] = 0,
    lr_scheduler: Annotated[
        str, Option(help="Learning rate scheduler type")
    ] = "constant_with_warmup",
    lr_scheduler_kwargs: Annotated[
        str, Option(help="JSON string of scheduler-specific kwargs")
    ] = "{}",
    seed: Annotated[int, Option(help="Random seed for reproducibility")] = 67,
    # AdamW optimizer parameters
    beta1: Annotated[
        float, Option(help="AdamW beta1 parameter (momentum coefficient)")
    ] = 0.9,
    beta2: Annotated[
        float, Option(help="AdamW beta2 parameter (RMSprop coefficient)")
    ] = 0.95,
    eps: Annotated[float, Option(help="AdamW epsilon for numerical stability")] = 1e-8,
    weight_decay: Annotated[
        float, Option(help="AdamW weight decay (L2 penalty)")
    ] = 0.0,
    use_liger_kernels: Annotated[
        bool, Option(help="Whether to use Liger kernels")
    ] = False,
    osft: Annotated[
        bool, Option(help="Enable OSFT (Orthogonal Subspace Fine-Tuning)")
    ] = False,
    osft_unfreeze_rank_ratio: Annotated[
        float,
        Option(
            help="Ratio of ranks to unfreeze for OSFT (0.0 = freeze all, 1.0 = unfreeze all). Required when osft is True"
        ),
    ] = None,
    osft_target_patterns: Annotated[
        str,
        Option(
            help=(
                "List of target modules to use for OSFT. When not provided, it will try to guess the patterns based on the model. "
                "This should be a comma-separated list of patterns. "
                "For example, 'self_attn.q_proj,self_attn.k_proj,mlp.gate_proj'"
            )
        ),
    ] = None,
    osft_upcast_dtype: Annotated[
        str | None,
        Option(
            help="Upcast dtype for OSFT computations. Can be 'float16', 'bfloat16', 'float32', etc."
        ),
    ] = "float32",
    osft_output_dtype: Annotated[
        str | None,
        Option(
            help="Output dtype for OSFT. If None, uses original model dtype. Can be 'float16', 'bfloat16', 'float32', etc."
        ),
    ] = None,
    osft_memory_efficient_init: Annotated[
        bool,
        Option(
            help=(
                "DEPRECATED: This flag is now ignored and will be removed in v0.5.0. "
                "Memory-efficient initialization is automatically enabled for distributed training. "
                "This parameter has no effect and can be safely removed."
            )
        ),
    ] = False,
    output_dir: Annotated[
        str, Option(help="Directory to save checkpoints and logs (required)")
    ] = ...,
    min_samples_per_checkpoint: Annotated[
        int | None,
        Option(
            help="Minimum number of samples processed before saving a checkpoint (required)"
        ),
    ] = None,
    # Training mode parameters
    training_mode: Annotated[
        TrainingMode,
        Option(
            help="Training mode: epoch, step, token, or infinite", case_sensitive=False
        ),
    ] = TrainingMode.EPOCH,
    max_epochs: Annotated[
        int, Option(help="Maximum number of epochs (for epoch mode)")
    ] = 1,
    max_steps: Annotated[
        int, Option(help="Maximum number of steps (for step mode)")
    ] = 0,
    max_tokens: Annotated[
        int, Option(help="Maximum number of loss-counted tokens (for token mode)")
    ] = 0,
    checkpoint_at_epoch: Annotated[
        bool, Option(help="Whether to save checkpoints at the end of each epoch")
    ] = False,
    save_final_checkpoint: Annotated[
        bool, Option(help="Whether to save a final checkpoint when training ends")
    ] = False,
    save_dtype: Annotated[
        str | None,
        Option(
            help="Dtype to save the model in. If None, uses original model dtype. Can be 'float16', 'bfloat16', 'float32', etc."
        ),
    ] = None,
    train_dtype: Annotated[
        str,
        Option(
            help="Dtype for training computations. Defaults to 'float32'. Can be 'float16', 'bfloat16', 'float32', etc."
        ),
    ] = "float32",
    # validation parameters
    validation_split: Annotated[
        float, Option(help="Fraction of data to use for validation (0.0 to 1.0)")
    ] = 0.0,
    validation_frequency: Annotated[
        int | None,
        Option(
            help="Frequency of validation evaluation (in steps). Required when validation_split > 0"
        ),
    ] = None,
    # checkpoint parameters
    save_best_val_loss: Annotated[
        bool, Option(help="Whether to save checkpoints when validation loss improves")
    ] = False,
    val_loss_improvement_threshold: Annotated[
        float,
        Option(help="Minimum validation loss improvement required to trigger a save"),
    ] = 0.0,
    # pretraining parameters
    block_size: Annotated[
        int | None,
        Option(
            help="Block size for pretraining mode (in tokens). When provided, enables pretraining with block-based sampling"
        ),
    ] = None,
    # wandb parameters
    wandb_project: Annotated[
        str | None, Option(help="Weights & Biases project name")
    ] = None,
    wandb_run_name: Annotated[
        str | None, Option(help="Weights & Biases run name")
    ] = None,
    wandb_entity: Annotated[
        str | None, Option(help="Weights & Biases entity/team name")
    ] = None,
    # mlflow parameters
    mlflow_tracking_uri: Annotated[
        str | None, Option(help="MLflow tracking server URI")
    ] = None,
    mlflow_experiment_name: Annotated[
        str | None, Option(help="MLflow experiment name")
    ] = None,
    mlflow_run_name: Annotated[str | None, Option(help="MLflow run name")] = None,
):
    # Reproducibility: align with HF Trainer seeding behavior
    set_seed(seed)

    init_distributed_environment()
    # TODO: make the path creation lazy, but confirm that we can write to the given directory
    # at this point
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # validation, do this before continuing execution flow so we don't log experiments that are invalid from
    # the get-go
    if osft:
        if osft_unfreeze_rank_ratio is None:
            raise ValueError("osft_unfreeze_rank_ratio is required when osft is True")
        if osft_target_patterns:
            osft_target_patterns = (
                osft_target_patterns.replace("'", "")
                .replace('"', "")
                .replace(" ", "")
                .split(",")
            )

        # Deprecation warning for osft_memory_efficient_init
        if osft_memory_efficient_init:
            import warnings

            warnings.warn(
                "The 'osft_memory_efficient_init' parameter is deprecated and will be "
                "removed in mini_trainer v0.5.0. Memory-efficient initialization is now "
                "automatically enabled for distributed training (when torch.distributed is "
                "initialized). This flag no longer has any effect and can be safely removed "
                "from your training configuration.",
                DeprecationWarning,
                stacklevel=2,
            )

    # TODO(osilkin): we should eventually put this validation logic somewhere dedicated but
    # for now it's easy to read here
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0 (exclusive)")

    if validation_split > 0.0 and (
        validation_frequency is None or validation_frequency <= 0
    ):
        raise ValueError(
            "validation_frequency must be provided and positive when validation_split > 0"
        )

    # Convert string dtypes to torch dtypes
    osft_upcast_dtype_torch = parse_dtype(osft_upcast_dtype)
    osft_output_dtype_torch = parse_dtype(osft_output_dtype)
    train_dtype_torch = parse_dtype(train_dtype)

    # Initialize logging flags
    use_wandb = wandb_project is not None
    use_mlflow = any([mlflow_tracking_uri, mlflow_experiment_name, mlflow_run_name])

    # Log parameters only on rank 0
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    node_rank = get_node_rank()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    if local_rank == 0:
        params = {
            "model_name_or_path": model_name_or_path,
            "data_path": data_path,
            "batch_size": batch_size,
            "max_tokens_per_gpu": max_tokens_per_gpu,
            "learning_rate": learning_rate,
            "num_warmup_steps": num_warmup_steps,
            "lr_scheduler": lr_scheduler,
            "seed": seed,
            "use_liger_kernels": use_liger_kernels,
            "osft": osft,
            "osft_unfreeze_rank_ratio": osft_unfreeze_rank_ratio,
            "osft_target_patterns": osft_target_patterns,
            "osft_upcast_dtype": osft_upcast_dtype,
            "osft_output_dtype": osft_output_dtype,
            "osft_memory_efficient_init": osft_memory_efficient_init,
            "output_dir": output_dir,
            "min_samples_per_checkpoint": min_samples_per_checkpoint,
            "save_dtype": save_dtype,
            "train_dtype": train_dtype,
            "training_mode": training_mode.value,
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "checkpoint_at_epoch": checkpoint_at_epoch,
            "save_final_checkpoint": save_final_checkpoint,
            "validation_split": validation_split,
            "validation_frequency": validation_frequency,
            "save_best_val_loss": save_best_val_loss,
            "val_loss_improvement_threshold": val_loss_improvement_threshold,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_entity": wandb_entity,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "LOCAL_RANK": local_rank,
            "GLOBAL_RANK": global_rank,
            "NODE_RANK": node_rank,
            "WORLD_SIZE": world_size,
        }

        # Initialize wandb with the same params config
        if use_wandb:
            # we rely on the WANDB_API_KEY being set as our primary mechanism for
            # authentication. So we error out here if it was requested but the user
            # is not authenticated
            if os.environ.get("WANDB_API_KEY") is None:
                raise ValueError(
                    "WANDB_API_KEY is not set. Please set the WANDB_API_KEY environment variable."
                )
            wandb_wrapper.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config=params,
            )
            log_rank_0(f"Initialized wandb project: {wandb_project}")

        # Initialize mlflow with the same params config
        # Only init on global rank 0 to avoid multiple runs in multi-node setups
        if use_mlflow and global_rank == 0:
            mlflow_wrapper.init(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=mlflow_experiment_name,
                run_name=mlflow_run_name
                or wandb_run_name,  # fallback to wandb_run_name
            )
            # Log hyperparameters
            mlflow_wrapper.log_params(params)
            log_rank_0(f"Initialized mlflow with tracking URI: {mlflow_tracking_uri}")

        params_path = output_path / "training_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        # Pretty print parameters in a single line using JSON
        print(
            f"Training with parameters: {json.dumps(params, separators=(',', ':'), indent=4)}"
        )
        print(f"Training parameters saved to {params_path}")

    setup_logger(level="INFO")

    # Parse scheduler kwargs from JSON string
    try:
        scheduler_kwargs_dict = (
            json.loads(lr_scheduler_kwargs) if lr_scheduler_kwargs else {}
        )
    except json.JSONDecodeError:
        log_rank_0(
            f"Warning: Invalid JSON for lr_scheduler_kwargs: {lr_scheduler_kwargs}. Using empty dict."
        )
        scheduler_kwargs_dict = {}

    # If Orthogonal Subspace Learning is enabled, loads a model with decomposed trainable low-rank + fixed high-rank subspace weights (see osft_utils)
    # Convert user-facing osft_unfreeze_rank_ratio to internal osft_rank_ratio
    osft_rank_ratio = (
        None if osft_unfreeze_rank_ratio is None else (1.0 - osft_unfreeze_rank_ratio)
    )
    model = setup_model(
        model_name_or_path=model_name_or_path,
        save_dtype=save_dtype,
        train_dtype=train_dtype_torch,
        use_liger_kernels=use_liger_kernels,
        osft=osft,
        local_rank=local_rank,
        osft_rank_ratio=osft_rank_ratio,
        osft_target_patterns=osft_target_patterns,
        osft_upcast_dtype=osft_upcast_dtype_torch,
        osft_output_dtype=osft_output_dtype_torch,
    )

    # Create PretrainingConfig if block_size is provided
    pretraining_config = None
    if block_size is not None:
        pretraining_config = PretrainingConfig(block_size=block_size)
        log_rank_0(f"Pretraining mode enabled with block_size={block_size}")

    # grab the data loader prior to the model so we can extract the dataset length
    # and use this for calculating the number of training steps in the data loader
    data_loader, val_data_loader = get_data_loader(
        data_path=data_path,
        batch_size=batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        seed=seed,
        pad_token_id=model.config.pad_token_id,
        validation_split=validation_split,
        pretraining_config=pretraining_config,
    )

    if validation_split > 0.0:
        log_rank_0(
            f"Created train/validation split with {validation_split:.1%} validation data"
        )
        log_rank_0(f"Validation data loader length: {len(val_data_loader)}")
        log_rank_0(f"Training data loader length: {len(data_loader)}")
    else:
        log_rank_0("No validation split - using all data for training")
        log_rank_0(f"Training data loader length: {len(data_loader)}")

    # Calculate number of training steps based on training mode
    num_training_steps = calculate_num_training_steps(
        training_mode=training_mode,
        data_loader=data_loader,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )

    log_rank_0(f"Calculated num_training_steps: {num_training_steps}")
    model, optimizer, lr_scheduler = setup_training_components(
        model=model,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps,
        lr_scheduler=lr_scheduler,
        num_training_steps=num_training_steps,
        scheduler_kwargs=scheduler_kwargs_dict,
        # AdamW optimizer parameters
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    train(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_loader=data_loader,
        output_dir=output_dir,
        min_samples_per_checkpoint=min_samples_per_checkpoint,
        model_name_or_path=model_name_or_path,
        training_mode=training_mode,
        max_epochs=max_epochs,
        max_steps=max_steps,
        max_tokens=max_tokens,
        checkpoint_at_epoch=checkpoint_at_epoch,
        save_final_checkpoint=save_final_checkpoint,
        train_dtype=train_dtype_torch,
        save_best_val_loss=save_best_val_loss,
        val_loss_improvement_threshold=val_loss_improvement_threshold,
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
        val_data_loader=val_data_loader,
        validation_frequency=validation_frequency,
    )

    # once done, tear down distributed environment
    if use_wandb:
        wandb_wrapper.finish()
    # Only finish mlflow on global rank 0 (where we started it)
    if use_mlflow and torch.distributed.get_rank() == 0:
        mlflow_wrapper.finish()
    destroy_distributed_environment()


if __name__ == "__main__":
    app()


"""
rclone copy --copy-links /new_data/experiments_rh/phi-4_limo_trainer_pipe_cleaner/hf_format/samples_8192.0 /dev/shm/phi-4_limo_trainer_pipe_cleaner_cont
        --data-path /dev/shm/knowledge_processed.jsonl \
        --data-path ./some_product_puzzle_tokenized_qwen1.5b.jsonl \
        --data-path ./mihir_prob.jsonl \
        --output-dir /new_data/experiments_rh/mihir_prob_qwen1.5b_v2     \
torchrun --nnodes=1 --nproc-per-node=8 train.py \
        --output-dir /cloud/misc/aldo/experiment/qwen32b-expert-iteration-test \
        --data-path ./tokenized_data.jsonl \
        --model-name-or-path Qwen/Qwen3-32B \
        --min-samples-per-checkpoint 3400 \
        --num-warmup-steps 20 \
        --max-tokens-per-gpu 60000              \
        --batch-size 128                       \
        --use-liger-kernels                    \
        --seed 893                               \
        --fsdp-sharding-strategy FULL_SHARD \
        --learning-rate 6e-6
"""
