"""
Pytest configuration and fixtures for the test suite.

This file provides shared fixtures and configurations for all tests,
particularly handling multiprocessing context for DataLoader tests.
"""

import sys

import pytest
import torch
import torch.multiprocessing as mp


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "gpu: requires a real accelerator, CUDA or XPU (deselect with '-m \"not gpu\"')")
    # Set multiprocessing start method to 'spawn' for better compatibility
    # This is crucial for avoiding segmentation faults in DataLoader tests
    if sys.platform != "win32":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, that's fine
            pass

    # Ensure CUDA operations are deterministic for testing
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture(autouse=True)
def pin_cuda_code_path_for_unit_tests(request, monkeypatch):
    """Keep CPU unit tests on the CUDA code path, as on a CPU-only CI host.

    These tests mock ``torch.cuda`` and expect it to be used. On an Intel XPU
    host mini_trainer would otherwise select XPU. gpu_tests see the real device.
    """
    if "gpu_tests" not in request.node.nodeid:
        monkeypatch.setenv("MINI_TRAINER_DEVICE", "cuda")


@pytest.fixture(autouse=True)
def cleanup_multiprocessing():
    """
    Automatically clean up multiprocessing resources after each test.

    This fixture runs after each test to ensure that any DataLoader
    workers are properly terminated and resources are freed.
    """
    yield

    # Force garbage collection to clean up any lingering DataLoader objects
    import gc

    gc.collect()

    # If CUDA is available, clear the cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Small delay to allow processes to fully terminate
    import time

    time.sleep(0.1)


@pytest.fixture
def temp_data_file(tmp_path):
    """
    Create a temporary JSONL data file for testing.

    This fixture creates a temporary directory and JSONL file
    that can be used for DataLoader tests.
    """
    import json

    data_file = tmp_path / "test_data.jsonl"
    samples = []

    for i in range(100):
        samples.append(
            {
                "input_ids": list(range(i, i + 10)),
                "labels": list(range(i, i + 10)),
                "len": 10,
                "num_loss_counted_tokens": 10,
            }
        )

    with open(data_file, "w") as f:
        for sample in samples:
            json.dump(sample, f)
            f.write("\n")

    return str(data_file)


@pytest.fixture
def zero_workers_dataloader(monkeypatch):
    """
    Fixture to force DataLoader to use zero workers.

    This can be used for tests that have issues with multiprocessing
    to ensure they run with num_workers=0.
    """
    import mini_trainer.sampler as sampler

    original_get_data_loader = sampler.get_data_loader

    def patched_get_data_loader(**kwargs):
        # Force num_workers to 0
        return sampler.DataLoader(
            sampler.JsonlDataset(kwargs["data_path"]),
            kwargs["batch_size"],
            sampler=sampler.InfiniteSampler(len(sampler.JsonlDataset(kwargs["data_path"])), seed=kwargs["seed"]),
            collate_fn=sampler.MaxTokensPerRankCollator(
                kwargs["max_tokens_per_gpu"],
                rank=kwargs.get("rank"),
                world_size=kwargs.get("world_size"),
                dummy_sample=kwargs.get("dummy_sample"),
            ),
            num_workers=0,  # Force to 0 to avoid multiprocessing issues
        )

    monkeypatch.setattr(sampler, "get_data_loader", patched_get_data_loader)
    yield
    # Cleanup happens automatically when monkeypatch is torn down
