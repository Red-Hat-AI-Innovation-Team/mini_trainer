"""
Shared type definitions and dataclasses for mini_trainer.

This module consolidates all common type definitions, enums, and dataclasses
used across the mini_trainer package to avoid duplication and ensure consistency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Literal


class TrainingMode(str, Enum):
    """Training mode determines the stopping criterion for training."""
    EPOCH = "epoch"
    STEP = "step"
    TOKEN = "token"
    INFINITE = "infinite"


@dataclass
class TorchrunArgs:
    """Arguments for torchrun distributed training configuration."""
    nnodes: int = 1
    nproc_per_node: Literal["gpu"] | int = 1
    node_rank: int = 0
    rdzv_id: str | int = 123

    # Optional rendezvous / master fields
    rdzv_endpoint: Optional[str] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None

    def __post_init__(self):
        # in order to support systems which are still relying on `master_addr`
        # to construct the rendezvous address, torchrun must not be given a non-empty value
        # for rdzv_endpoint:
        # https://github.com/pytorch/pytorch/blob/ecb53078faf86ca1b33277df33b82985675bb011/torch/distributed/run.py#L799
        if self.rdzv_endpoint and self.master_addr: 
            raise ValueError(
                "Provide either `rdzv_endpoint` OR both `master_addr` and `master_port`, not both."
            )

@dataclass
class TrainingArgs:
    """Complete training configuration arguments."""
    # Required fields (no defaults)
    model_name_or_path: str = field(metadata={"help": "The name or path of the model to train."})
    data_path: str = field(metadata={"help": "The path to the training data."})
    batch_size: int = field(metadata={"help": "The batch size to use for training."})
    max_tokens_per_gpu: int = field(metadata={"help": "The maximum number of tokens per GPU per minibatch."})
    learning_rate: float = field(metadata={"help": "The learning rate to use for training."})
    output_dir: str = field(metadata={"help": "Directory to save checkpoints and logs."})
    
    # Optional fields (with defaults)
    num_warmup_steps: int = field(default=0, metadata={"help": "The number of warmup steps for the learning rate scheduler."})
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "The learning rate scheduler to use. NOTE: Infinite mode only supports schedulers which do not read the number of training steps."},
    )
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict, metadata={"help": "Additional keyword arguments for the learning rate scheduler."})
    seed: int = field(default=42, metadata={"help": "The random seed to use for training."})
    
    # Model configuration
    use_liger_kernels: bool = field(default=False, metadata={"help": "Whether to use Liger kernels."})
    osft: bool = field(default=False, metadata={"help": "Whether to use OSFT (Orthogonal Subspace Fine-Tuning). If enabled, you must also specify the `osft_unfreeze_rank_ratio`."})
    osft_unfreeze_rank_ratio: float | None = field(default=None, metadata={"help": "The ratio of ranks that will be unfrozen in each weight matrix during OSFT. 0.0 means the entire matrix is frozen, 0.2 means the 20% smallest singular values will be unfrozen, and 1.0 means the entire matrix is unfrozen."})
    osft_target_patterns: list[str] | None = field(default=None, metadata={
        "help": "A list of patterns to match against the model's parameter names to target for OSFT. By default, we try to resolve the configuration which best suits your model.",
    })
    osft_upcast_dtype: str | None = field(default="float32", metadata={"help": "Upcast dtype for OSFT computations. Can be 'float16', 'bfloat16', 'float32', etc."})
    osft_output_dtype: str | None = field(default=None, metadata={"help": "Output dtype for OSFT. If None, uses original model dtype. Can be 'float16', 'bfloat16', 'float32', etc."})
    osft_memory_efficient_init: bool = field(default=False, metadata={"help": "Enable memory-efficient OSFT initialization (useful for large models and GPT-OSS)"})
    
    # Output options
    min_samples_per_checkpoint: Optional[int] = field(default=None, metadata={"help": "If provided, this must be the number of samples to process before saving a checkpoint."})
    
    # Training mode and stopping criteria
    training_mode: TrainingMode = field(default=TrainingMode.EPOCH, metadata={
        "help": (
            "The training mode to use.\n"
            "EPOCH: Train for a fixed number of epochs.\n"
            "STEP: Train for a fixed number of steps.\n"
            "TOKEN: Train for a fixed number of tokens.\n"
            "INFINITE: Train indefinitely until the user stops the process.\n"
            "NOTE: Infinite mode only supports schedulers which do not read the number of training steps.\n"
        )}
    )
    max_epochs: int = 1  # For EPOCH mode
    max_steps: int = 0   # For STEP mode  
    max_tokens: int = 0  # For TOKEN mode
    
    # Checkpointing
    checkpoint_at_epoch: bool = field(default=False, metadata={"help": "Whether to checkpoint at the end of each epoch."})
    save_final_checkpoint: bool = field(default=True, metadata={"help": "Whether the model should be saved at the end of training or not. Off by default to avoid accidentally overwriting the best checkpoint."})
    save_dtype: str | None = field(default=None, metadata={"help": "The dtype to save the model in. If None, uses original model dtype. Can be 'float16', 'bfloat16', 'float32', etc."})
    train_dtype: str = field(default="float32", metadata={"help": "Dtype for training computations. Can be 'float16', 'bfloat16', 'float32', etc."})

    # Weights & Biases integration
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Weights & Biases project name."})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Weights & Biases run name."})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Weights & Biases entity/team name."})

    # validation
    validation_split: float = field(default=0.0, metadata={"help": "The fraction of data to use for validation. 0.0 means no validation, 0.1 means 10% of the data is used for validation."})
    validation_frequency: Optional[int] = field(default=None, metadata={"help": "The frequency of validation in steps. Required when validation_split > 0."})

    # from train.py:
    save_best_val_loss: bool = field(default=False, metadata={"help": "Whether to save checkpoints when validation loss improves"})
    val_loss_improvement_threshold: float = field(default=0.0, metadata={"help": "Minimum validation loss improvement required to trigger a save"})
