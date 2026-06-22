"""Mini Trainer - A simple training library for PyTorch models.

This package provides reference implementations of emerging training algorithms,
including Orthogonal Subspace Fine Tuning (OSFT).
"""

# Dynamic version from setuptools_scm
try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations
    __version__ = "unknown"


from . import (
    api_train,
    batch_metrics,
    batch_packer,
    callbacks,
    none_reduction_losses,
    osft_utils,
    sampler,
    setup_model_for_training,
    train,
    utils,
)

# Export main API functions for convenience
from .api_train import run_training
from .callbacks import CallbackManager, TrainerCallback, TrainingContext
from .training_types import PretrainingConfig, TorchrunArgs, TrainingArgs, TrainingMode

__all__ = [
    "api_train",
    "batch_metrics",
    "batch_packer",
    "callbacks",
    "none_reduction_losses",
    "sampler",
    "setup_model_for_training",
    "osft_utils",
    "train",
    "utils",
    # Main API exports
    "run_training",
    "TorchrunArgs",
    "TrainingArgs",
    "TrainingMode",
    "PretrainingConfig",
    "CallbackManager",
    "TrainerCallback",
    "TrainingContext",
]
