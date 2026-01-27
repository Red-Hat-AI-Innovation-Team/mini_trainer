# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for optional mlflow imports that provides consistent error handling
across all processes when mlflow is not installed.
"""

import logging
from typing import Any, Dict, Optional

# Try to import mlflow
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

# Store the active run ID to ensure we log to the correct run
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

    Args:
        tracking_uri: MLflow tracking server URI (e.g., "http://localhost:5000")
        experiment_name: Name of the experiment
        run_name: Name of the run
        **kwargs: Additional arguments to pass to mlflow.start_run

    Returns:
        mlflow.ActiveRun object if successful

    Raises:
        MLflowNotAvailableError: If mlflow is not installed
    """
    global _active_run_id
    check_mlflow_available("initialize mlflow")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name, **kwargs)
    _active_run_id = run.info.run_id
    return run


def get_active_run_id() -> Optional[str]:
    """Get the active run ID that was started by init()."""
    return _active_run_id


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
    # Use the stored run ID to ensure we log to the correct run
    if _active_run_id:
        with mlflow.start_run(run_id=_active_run_id):
            mlflow.log_params(str_params)
    else:
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
        # Use the stored run ID to ensure we log to the correct run
        # This is critical for async logging where thread-local context may be lost
        if _active_run_id:
            with mlflow.start_run(run_id=_active_run_id):
                mlflow.log_metrics(metrics, step=step)
        else:
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
