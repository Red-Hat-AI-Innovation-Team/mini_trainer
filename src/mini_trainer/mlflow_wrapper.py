# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for optional mlflow imports that provides consistent error handling
across all processes when mlflow is not installed.
"""

import logging
import os
from typing import Any, Dict, Optional

# Try to import mlflow
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

# Store the active run ID to ensure we can resume the run if needed
# This is needed because async logging may lose the thread-local run context
_active_run_id: Optional[str] = None


class MLflowNotAvailableError(ImportError):
    """Raised when mlflow functions are called but mlflow is not installed."""

    pass


def check_mlflow_available(operation: str) -> None:
    """Check if mlflow is available, raise error if not."""
    if not MLFLOW_AVAILABLE:
        error_msg = (
            f"Attempted to {operation} but mlflow is not installed. "
            "Please install mlflow with: pip install mlflow"
        )
        logger.error(error_msg)
        raise MLflowNotAvailableError(error_msg)


def init(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Initialize an mlflow run. Raises MLflowNotAvailableError if mlflow is not installed.

    Configuration follows a precedence hierarchy:
        1. Explicit kwargs (highest priority)
        2. Environment variables (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
        3. MLflow defaults (lowest priority)

    Args:
        tracking_uri: MLflow tracking server URI (e.g., "http://localhost:5000").
            Falls back to MLFLOW_TRACKING_URI environment variable if not provided.
        experiment_name: Name of the experiment.
            Falls back to MLFLOW_EXPERIMENT_NAME environment variable if not provided.
        run_name: Name of the run
        **kwargs: Additional arguments to pass to mlflow.start_run

    Returns:
        mlflow.ActiveRun object if successful

    Raises:
        MLflowNotAvailableError: If mlflow is not installed
    """
    global _active_run_id
    check_mlflow_available("initialize mlflow")

    # Apply kwarg > env var precedence for tracking_uri
    effective_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if effective_tracking_uri:
        mlflow.set_tracking_uri(effective_tracking_uri)

    # Apply kwarg > env var precedence for experiment_name
    effective_experiment_name = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT_NAME"
    )
    if effective_experiment_name:
        mlflow.set_experiment(effective_experiment_name)

    # Remove run_name from kwargs if present to avoid duplicate keyword argument
    # The explicit run_name parameter takes precedence
    kwargs.pop("run_name", None)

    # Reuse existing active run if one exists, otherwise start a new one
    active_run = mlflow.active_run()
    if active_run is not None:
        run = active_run
    else:
        run = mlflow.start_run(run_name=run_name, **kwargs)
    _active_run_id = run.info.run_id
    return run


def get_active_run_id() -> Optional[str]:
    """Get the active run ID that was started by init()."""
    return _active_run_id


def _ensure_run_for_logging() -> None:
    """Ensure there's an active MLflow run for logging.

    This helper handles async contexts where thread-local run context may be lost.
    If no active run exists but we have a stored run ID, it resumes that run.
    """
    active_run = mlflow.active_run()
    if not active_run and _active_run_id:
        # No active run in this thread but we have a stored run ID - resume it
        # This can happen in async contexts where thread-local context is lost
        # Note: We don't use context manager here because it would end the run on exit
        mlflow.start_run(run_id=_active_run_id)


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to mlflow. Raises MLflowNotAvailableError if mlflow is not installed.

    Args:
        params: Dictionary of parameters to log

    Raises:
        MLflowNotAvailableError: If mlflow is not installed
    """
    check_mlflow_available("log params to mlflow")
    # MLflow params must be strings
    str_params = {k: str(v) for k, v in params.items()}

    _ensure_run_for_logging()
    mlflow.log_params(str_params)


def log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to mlflow. Raises MLflowNotAvailableError if mlflow is not installed.

    Args:
        data: Dictionary of data to log (non-numeric values will be skipped)
        step: Optional step number for the metrics

    Raises:
        MLflowNotAvailableError: If mlflow is not installed
    """
    check_mlflow_available("log to mlflow")
    # Filter to only numeric values for metrics
    metrics = {}
    for k, v in data.items():
        try:
            metrics[k] = float(v)
        except (ValueError, TypeError):
            pass  # Skip non-numeric values
    if metrics:
        _ensure_run_for_logging()
        mlflow.log_metrics(metrics, step=step)


def finish() -> None:
    """
    End the mlflow run. Raises MLflowNotAvailableError if mlflow is not installed.

    Raises:
        MLflowNotAvailableError: If mlflow is not installed
    """
    global _active_run_id
    check_mlflow_available("finish mlflow run")
    mlflow.end_run()
    _active_run_id = None
