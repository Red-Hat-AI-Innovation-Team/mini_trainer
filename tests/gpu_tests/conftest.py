"""Pytest configuration for GPU tests."""

import sys
from pathlib import Path

import pytest
import torch

from mini_trainer import device_utils

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Add custom markers for GPU tests."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')")


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_count():
    """Return number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


@pytest.fixture(scope="session")
def flash_attn_available():
    """Check if flash attention is available."""
    try:
        import flash_attn as _  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_flash_attn(flash_attn_available):
    """Skip test if flash attention is not available."""
    if not flash_attn_available:
        pytest.skip("Flash attention not available")


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset accelerator (CUDA/XPU) memory before and after each test."""
    if device_utils.is_accelerator_available():
        device_utils.empty_cache()
        device_utils.reset_peak_memory_stats()
    yield
    if device_utils.is_accelerator_available():
        device_utils.empty_cache()


@pytest.fixture
def single_gpu_device():
    """Provide a single accelerator (CUDA or XPU) device for testing."""
    if not device_utils.is_accelerator_available():
        pytest.skip("No CUDA or XPU accelerator available")
    return device_utils.get_device(0)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return str(checkpoint_dir)
