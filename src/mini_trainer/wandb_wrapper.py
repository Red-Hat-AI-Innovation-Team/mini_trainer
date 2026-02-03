# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for optional wandb imports that provides consistent error handling 
across all processes when wandb is not installed.
"""

import logging
from typing import Any, Dict, Optional

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class WandbNotAvailableError(ImportError):
    """Raised when wandb functions are called but wandb is not installed."""
    pass


def check_wandb_available(operation: str) -> None:
    """Check if wandb is available, raise error if not."""
    if not WANDB_AVAILABLE:
        error_msg = (
            f"Attempted to {operation} but wandb is not installed. "
            "Please install wandb with: pip install wandb"
        )
        logger.error(error_msg)
        raise WandbNotAvailableError(error_msg)


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Initialize a wandb run. Raises WandbNotAvailableError if wandb is not installed.
    
    Args:
        project: Name of the project
        name: Name of the run
        entity: Entity/team name
        config: Configuration dictionary
        **kwargs: Additional arguments to pass to wandb.init
        
    Returns:
        wandb.Run object if successful
        
    Raises:
        WandbNotAvailableError: If wandb is not installed
    """
    check_wandb_available("initialize wandb")
    return wandb.init(
        project=project,
        name=name,
        entity=entity,
        config=config,
        **kwargs
    )


def log(data: Dict[str, Any], **kwargs) -> None:
    """
    Log data to wandb. Raises WandbNotAvailableError if wandb is not installed.
    
    Args:
        data: Dictionary of data to log
        **kwargs: Additional arguments to pass to wandb.log
        
    Raises:
        WandbNotAvailableError: If wandb is not installed
    """
    check_wandb_available("log to wandb")
    wandb.log(data, **kwargs)


def finish() -> None:
    """
    Finish the wandb run. Raises WandbNotAvailableError if wandb is not installed.
    
    Raises:
        WandbNotAvailableError: If wandb is not installed
    """
    check_wandb_available("finish wandb run")
    wandb.finish()


