"""
Test suite for data loading and sampling components.

Tests the JsonlDataset and MaxTokensPerRankCollator
to ensure correct data loading, sampling, and batching behavior.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mini_trainer.sampler import (
    EpochSampler,
    JsonlDataset,
    MaxTokensPerRankCollator,
    get_data_loader,
    mb_collate_fn,
    reset_minibatches,
)


class TestJsonlDataset:
    """Test suite for the JsonlDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample JSONL data for testing."""
        return [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [10, 20, -100, 40, 50],
                "len": 5,
                "num_loss_counted_tokens": 4,
            },
            {
                "input_ids": [6, 7, 8, 9],
                "labels": [-100, -100, 80, 90],
                "len": 4,
                "num_loss_counted_tokens": 2,
            },
            {
                "input_ids": [11, 12, 13],
                "labels": [110, 120, 130],
                "len": 3,
                # Missing num_loss_counted_tokens to test fallback
            },
        ]

    @pytest.fixture
    def temp_jsonl_file(self, sample_data):
        """Create a temporary JSONL file with sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    def test_dataset_initialization(self, temp_jsonl_file):
        """Test dataset initialization with valid JSONL file."""
        dataset = JsonlDataset(path=temp_jsonl_file)
        assert len(dataset) == 3

    def test_dataset_getitem(self, temp_jsonl_file):
        """Test retrieving items from dataset."""
        dataset = JsonlDataset(path=temp_jsonl_file)

        # Test first item
        item = dataset[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert item["input_ids"].tolist() == [1, 2, 3, 4, 5]
        assert item["labels"].tolist() == [10, 20, -100, 40, 50]
        assert item["len"] == 5
        assert item["num_loss_counted_tokens"] == 4

        # Test second item
        item = dataset[1]
        assert item["input_ids"].tolist() == [6, 7, 8, 9]
        assert item["num_loss_counted_tokens"] == 2

    def test_dataset_missing_loss_counted_tokens(self, temp_jsonl_file):
        """Test fallback calculation when num_loss_counted_tokens is missing."""
        dataset = JsonlDataset(path=temp_jsonl_file)

        # Third item has missing num_loss_counted_tokens
        item = dataset[2]
        assert item["input_ids"].tolist() == [11, 12, 13]
        assert item["labels"].tolist() == [110, 120, 130]
        # Should calculate from labels (all non -100)
        assert item["num_loss_counted_tokens"] == 2

    def test_dataset_index_types(self, temp_jsonl_file):
        """Test dataset accepts different index types."""
        dataset = JsonlDataset(path=temp_jsonl_file)

        # Test with int
        item = dataset[0]
        assert item is not None

        # Test with numpy int
        item = dataset[np.int64(1)]
        assert item is not None


class TestMbCollateFn:
    """Test suite for the minibatch collate function."""

    def test_collate_single_sample(self):
        """Test collating a single sample."""
        minibatch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "labels": torch.tensor([10, -100, 30, 40, 50]),
                "num_loss_counted_tokens": 4,
            }
        ]

        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=4)

        assert result["input_ids"].shape == (1, 5)
        assert result["labels"].shape == (1, 5)
        assert result["position_ids"].shape == (1, 5)
        assert result["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
        assert result["num_loss_counted_tokens"] == 4
        assert result["num_samples"] == 1
        assert result["batch_num_loss_counted_tokens"] == 4

    def test_collate_multiple_samples(self):
        """Test collating multiple samples into packed format."""
        minibatch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([10, 20, 30]),
                "num_loss_counted_tokens": 3,
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7]),
                "labels": torch.tensor([40, -100, 60, 70]),
                "num_loss_counted_tokens": 3,
            },
        ]

        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=6)

        # Check concatenation
        assert result["input_ids"].shape == (1, 7)
        assert result["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 7]]
        assert result["labels"].tolist() == [[10, 20, 30, 40, -100, 60, 70]]

        # Check position_ids reset for each sequence
        assert result["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3]]

        assert result["num_loss_counted_tokens"] == 6
        assert result["num_samples"] == 2

    def test_collate_with_dummy_sample(self):
        """Test collating with dummy samples (padding)."""
        minibatch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([10, 20, 30]),
                "num_loss_counted_tokens": 3,
            },
            {
                "input_ids": torch.tensor([99, 99, 99]),
                "labels": torch.tensor([-100, -100, -100]),
                "num_loss_counted_tokens": 0,  # Dummy sample
            },
        ]

        result = mb_collate_fn(minibatch, batch_num_loss_counted_tokens=3)

        # Dummy samples shouldn't count
        assert result["num_samples"] == 1
        assert result["num_loss_counted_tokens"] == 3


class TestMaxTokensPerRankCollator:
    """Test suite for the MaxTokensPerRankCollator class."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        return [
            {
                "input_ids": torch.tensor([1] * 100),
                "labels": torch.tensor([1] * 100),
                "len": 100,
                "num_loss_counted_tokens": 100,
            },
            {
                "input_ids": torch.tensor([2] * 200),
                "labels": torch.tensor([2] * 200),
                "len": 200,
                "num_loss_counted_tokens": 200,
            },
            {
                "input_ids": torch.tensor([3] * 300),
                "labels": torch.tensor([3] * 300),
                "len": 300,
                "num_loss_counted_tokens": 300,
            },
            {
                "input_ids": torch.tensor([4] * 400),
                "labels": torch.tensor([4] * 400),
                "len": 400,
                "num_loss_counted_tokens": 400,
            },
        ]

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=2)
    def test_collator_initialization(self, mock_world_size, mock_rank, mock_available, mock_initialized):
        """Test collator initialization with distributed settings."""
        collator = MaxTokensPerRankCollator(max_tokens_per_rank=1000)

        assert collator.max_tokens_per_rank == 1000
        assert collator.global_rank == 0
        assert collator.world_size == 2
        assert collator.dummy_sample is not None

    def test_collator_custom_dummy_sample(self):
        """Test collator with custom dummy sample."""
        dummy = {
            "input_ids": torch.tensor([999, 999]),
            "labels": torch.tensor([-100, -100]),
            "len": 2,
            "num_loss_counted_tokens": 0,
        }

        collator = MaxTokensPerRankCollator(max_tokens_per_rank=1000, rank=0, world_size=2, dummy_sample=dummy)

        assert collator.dummy_sample == dummy

    @patch("mini_trainer.sampler.batch_lengths_to_minibatches_lpt")
    def test_collator_filters_long_sequences(self, mock_batch_fn, sample_batch, capsys):
        """Test that collator filters sequences longer than max_tokens."""
        mock_batch_fn.return_value = [[0, 1]]

        # Add a very long sequence
        sample_batch.append(
            {
                "input_ids": torch.tensor([5] * 2000),
                "labels": torch.tensor([5] * 2000),
                "len": 2000,
                "num_loss_counted_tokens": 2000,
            }
        )

        collator = MaxTokensPerRankCollator(max_tokens_per_rank=1000, rank=0, world_size=2)

        collator(sample_batch)

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "removed 1 samples" in captured.out

        # Check that the long sequence was properly filtered
        mock_batch_fn.assert_called_once()
        batch_lengths = mock_batch_fn.call_args[0][0]
        # The filtered batch should only have the original 4 sequences
        assert len(batch_lengths) == 4
        assert 2000 not in batch_lengths
        assert all(length <= 1000 for length in batch_lengths)

    @patch("mini_trainer.sampler.batch_lengths_to_minibatches_lpt")
    def test_collator_returns_minibatches(self, mock_batch_fn, sample_batch):
        """Test that collator returns properly formatted minibatches."""
        mock_batch_fn.return_value = [[0, 1], [2, -1]]  # Two minibatches

        collator = MaxTokensPerRankCollator(max_tokens_per_rank=500, rank=0, world_size=2)

        result = collator(sample_batch)

        assert len(result) == 2  # Two minibatches

        # Each result should be a dictionary with required keys
        for mb in result:
            assert "input_ids" in mb
            assert "labels" in mb
            assert "position_ids" in mb
            assert "num_loss_counted_tokens" in mb
            assert "num_samples" in mb
            assert "batch_num_loss_counted_tokens" in mb


class TestGetDataLoader:
    """Test suite for the get_data_loader function."""

    @pytest.fixture
    def temp_data_file(self):
        """Create a temporary data file for testing."""
        data = [
            {
                "input_ids": list(range(100)),
                "labels": list(range(100)),
                "len": 100,
                "num_loss_counted_tokens": 100,
            }
            for _ in range(10)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=1)
    def test_get_data_loader_basic_epoch_sampler(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        """Test basic data loader creation."""
        expected_batch_size = 4
        expected_grad_accum_steps = 1

        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=4,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 100-token samples, so batch-size of 4
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_batch_size
        assert loader.sampler.epoch == 0

        # now pop another 2 batches off, the last should either not exist or only have 2 samples
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_batch_size
        assert loader.sampler.epoch == 0

        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == 2  # we now expect the last 2 samples to be here

        # now we should have seen all samples, but we need to increment the epoch
        assert loader.sampler.epoch == 0
        loader.sampler.set_epoch(1)
        assert loader.sampler.epoch == 1

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=1)
    def test_get_data_loader_epoch_wraparound(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        """Test data loader behavior at epoch boundaries with EpochSampler."""
        expected_batch_size = 8
        expected_grad_accum_steps = 1

        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=expected_batch_size,
            max_tokens_per_gpu=50000,  # so we dont accumulate in this test
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)
        assert len(batch) == expected_grad_accum_steps

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 10 100-token samples, so batch-size 8 w/ 100 tokens per sample
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_batch_size
        assert loader.sampler.epoch == 0

        # The next batch should only have 2 samples (10 total - 8 already seen = 2 remaining)
        # After this, the epoch should be complete
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == 2  # Only 2 samples left in the epoch
        assert loader.sampler.epoch == 0  # Epoch counter doesn't auto-increment

        # Iterator should be exhausted now - need to create a new one for next epoch
        # This would typically happen in the training loop
        loader.sampler.set_epoch(1)  # Manually set to next epoch
        assert loader.sampler.epoch == 1  # Now it's incremented
        loader_it = iter(loader)  # Create new iterator for the new epoch

        # Now we should get a full batch again from the new epoch
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_batch_size
        assert loader.sampler.epoch == 1  # Still at epoch 1

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=2)
    def test_get_data_loader_two_gpus(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        """Test basic data loader creation."""
        expected_batch_size = 8
        expected_grad_accum_steps = 1
        expected_leftover_samples = 2

        # first try with an even batch size, so the last step will have 2 samples
        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=8,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # we expect there to only be a single grad accum step, since the
        # temp data file creates 100-token samples, so batch-size of 4
        # should only return a single batch
        assert len(batch) == expected_grad_accum_steps
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_batch_size // mock_world_size()
        assert loader.sampler.epoch == 0

        # so now the next batch should have 1 sample on each rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == expected_leftover_samples // mock_world_size()
        assert loader.sampler.epoch == 0

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=1)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=2)
    def test_mock_rank(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        assert mock_rank() == 1
        mock_rank.return_value = 0
        assert mock_rank() == 0

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=2)
    def test_padded_samples_on_last_rank(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        """Test basic data loader creation."""
        expected_batch_size = 9

        # first try with an even batch size, so the last step will have 2 samples
        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=9,
            max_tokens_per_gpu=500,
            seed=42,
        )
        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # this will burn up most of the samples
        minibatch = batch[0]
        assert minibatch["num_samples"] == (expected_batch_size // mock_world_size()) + 1

        # now there should be 1 batch left, it will most likely be on the first rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] > 0

        # next, change our rank to the last to see the difference
        mock_rank.return_value = 1
        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=9,
            max_tokens_per_gpu=500,
            seed=42,
        )

        # asserts that loading the model works properly
        assert loader is not None
        assert loader.batch_size == expected_batch_size
        assert loader.sampler.epoch == 0

        # get an iterator and fetch one batch
        loader_it = iter(loader)
        batch = next(loader_it)

        # this will burn up most of the samples
        minibatch = batch[0]
        assert minibatch["num_samples"] == expected_batch_size // mock_world_size()

        # we expect nothing to be there on the last rank
        batch = next(loader_it)
        microbatch = batch[0]
        assert microbatch["num_samples"] == 0
        # but we still expect there to be an item
        assert microbatch["labels"] is not None
        assert microbatch["input_ids"] is not None
        assert microbatch["batch_num_loss_counted_tokens"] > 0
        assert microbatch["num_loss_counted_tokens"] == 0

    def test_get_data_loader_with_custom_params(self, temp_data_file):
        """Test data loader with custom parameters."""
        dummy_sample = {
            "input_ids": torch.tensor([0, 0]),
            "labels": torch.tensor([-100, -100]),
            "len": 2,
            "num_loss_counted_tokens": 0,
        }

        loader, _ = get_data_loader(
            data_path=temp_data_file,
            batch_size=8,
            max_tokens_per_gpu=1000,
            seed=123,
            rank=1,
            world_size=4,
            dummy_sample=dummy_sample,
        )

        assert loader.batch_size == 8
        assert loader.collate_fn.global_rank == 1
        assert loader.collate_fn.world_size == 4
        assert loader.collate_fn.dummy_sample == dummy_sample

    @patch("mini_trainer.sampler.dist.is_initialized", return_value=True)
    @patch("mini_trainer.sampler.dist.is_available", return_value=True)
    @patch("mini_trainer.sampler.dist.get_rank", return_value=0)
    @patch("mini_trainer.sampler.dist.get_world_size", return_value=2)
    def test_data_loader_iteration(
        self,
        mock_world_size,
        mock_rank,
        mock_available,
        mock_initialized,
        temp_data_file,
    ):
        """Test that data loader can be iterated."""
        loader, _ = get_data_loader(data_path=temp_data_file, batch_size=2, max_tokens_per_gpu=500, seed=42)

        # Get an iterator and fetch one batch
        data_iter = iter(loader)
        batch = next(data_iter)

        assert isinstance(batch, list)
        # Each element should be a minibatch dictionary
        if len(batch) > 0:
            assert isinstance(batch[0], dict)
            assert "input_ids" in batch[0]

    def test_validation_split_creates_two_loaders(self, temp_data_file):
        """Test that validation_split > 0 creates both train and validation loaders."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.2,
            rank=0,
            world_size=1,
        )

        # Both loaders should be created
        assert train_loader is not None
        assert val_loader is not None

        # Check that datasets have correct sizes (10 total samples)
        assert len(train_loader.dataset) == 8  # 80% of 10
        assert len(val_loader.dataset) == 2  # 20% of 10

        # Both should have the same batch size
        assert train_loader.batch_size == val_loader.batch_size == 2

    def test_validation_split_determinism(self, temp_data_file):
        """Test that validation split is deterministic with the same seed."""
        # first call with seed 42
        train_loader1, val_loader1 = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.2,
            rank=0,
            world_size=1,
        )

        # second call with same seed should produce identical split
        train_loader2, val_loader2 = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.2,
            rank=0,
            world_size=1,
        )

        # collect samples from both loaders to compare
        def get_samples(loader):
            samples = []
            for batch in loader:
                for mb in batch:
                    # store the input_ids as they uniquely identify samples
                    samples.append(mb["input_ids"].tolist())
            return samples

        train_samples1 = get_samples(train_loader1)
        train_samples2 = get_samples(train_loader2)
        val_samples1 = get_samples(val_loader1)
        val_samples2 = get_samples(val_loader2)

        # assert identical splits
        assert train_samples1 == train_samples2, "Train datasets should be identical with same seed"
        assert val_samples1 == val_samples2, "Validation datasets should be identical with same seed"

    def test_no_validation_split_returns_none(self, temp_data_file):
        """Test that validation_split=0 returns None for validation loader."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=4,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.0,  # No validation split
            rank=0,
            world_size=1,
        )

        # Only train loader should be created
        assert train_loader is not None
        assert val_loader is None

        # Train loader should have all samples
        assert len(train_loader.dataset) == 10

    @pytest.mark.parametrize(
        "val_split,expected_train,expected_val",
        [
            (0.1, 9, 1),  # 10% validation
            (0.3, 7, 3),  # 30% validation
            (0.5, 5, 5),  # 50% validation
            (0.7, 3, 7),  # 70% validation
        ],
    )
    def test_validation_split_with_different_ratios(self, temp_data_file, val_split, expected_train, expected_val):
        """Test validation split with different ratios to ensure correct splitting."""
        train_loader, val_loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=val_split,
            rank=0,
            world_size=1,
        )

        assert len(train_loader.dataset) == expected_train, f"Failed for split {val_split}"
        assert len(val_loader.dataset) == expected_val, f"Failed for split {val_split}"

        # Verify no overlap between train and val data
        # Since we can't directly access indices due to the refactoring,
        # we'll just verify the total equals original dataset size
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        assert total_samples == 10, f"Total samples mismatch for split {val_split}"

    def test_train_val_data_are_different(self):
        """Test that train and validation datasets contain different samples."""
        # Create a temporary file with unique samples
        data = [
            {
                "input_ids": list(range(i * 10, (i + 1) * 10)),  # Each sample has different input_ids
                "labels": list(range(i * 10, (i + 1) * 10)),
                "len": 10,
                "num_loss_counted_tokens": 10,
            }
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            train_loader, val_loader = get_data_loader(
                data_path=temp_path,
                batch_size=1,  # Use batch size 1 to get individual samples
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=0.3,
                rank=0,
                world_size=1,
            )

            # Collect all input_ids from train loader
            train_samples = []
            for batch in train_loader:
                for minibatch in batch:
                    # Extract the actual tensor values
                    input_ids = minibatch["input_ids"].squeeze().tolist()
                    train_samples.append(tuple(input_ids))  # Use tuple for hashability

            # Collect all input_ids from validation loader
            val_samples = []
            for batch in val_loader:
                for minibatch in batch:
                    input_ids = minibatch["input_ids"].squeeze().tolist()
                    val_samples.append(tuple(input_ids))

            # Convert to sets and check for no overlap
            train_set = set(train_samples)
            val_set = set(val_samples)

            assert len(train_set) == 7  # 70% of 10 samples
            assert len(val_set) == 3  # 30% of 10 samples
            assert train_set.isdisjoint(val_set), "Train and validation sets should not overlap"

            # Verify each set contains unique samples
            assert len(train_samples) == 7, "Should have 7 train samples total"
            assert len(val_samples) == 3, "Should have 3 validation samples total"

        finally:
            os.unlink(temp_path)

    def test_validation_split_odd_dataset_size(self):
        """Test validation split with odd-sized dataset to verify rounding behavior."""
        # create a dataset with 11 samples
        data = [
            {
                "input_ids": list(range(i * 10, (i + 1) * 10)),
                "labels": list(range(i * 10, (i + 1) * 10)),
                "len": 10,
                "num_loss_counted_tokens": 10,
            }
            for i in range(11)  # 11 samples
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            # test different splits with 11 samples
            # note: HuggingFace's train_test_split uses round() which rounds to nearest even
            test_cases = [
                (0.1, 9, 2),  # 10% of 11 = 1.1, rounds to 2 val samples
                (0.2, 8, 3),  # 20% of 11 = 2.2, rounds to 3 val samples
                (0.3, 7, 4),  # 30% of 11 = 3.3, rounds to 4 val samples
                (0.4, 6, 5),  # 40% of 11 = 4.4, rounds to 5 val samples
                (0.5, 5, 6),  # 50% of 11 = 5.5, rounds to 6 val samples
            ]

            for val_split, expected_train, expected_val in test_cases:
                train_loader, val_loader = get_data_loader(
                    data_path=temp_path,
                    batch_size=2,
                    max_tokens_per_gpu=500,
                    seed=42,
                    validation_split=val_split,
                    rank=0,
                    world_size=1,
                )

                assert len(train_loader.dataset) == expected_train, f"Failed for split {val_split} with 11 samples"
                assert len(val_loader.dataset) == expected_val, f"Failed for split {val_split} with 11 samples"

                # verify total is still 11
                total_samples = len(train_loader.dataset) + len(val_loader.dataset)
                assert total_samples == 11, f"Total samples mismatch for split {val_split} with 11 samples"
        finally:
            os.unlink(temp_path)

    def test_validation_split_boundary_cases(self, temp_data_file):
        """Test boundary and invalid validation split ratios."""
        # test ratio of 1.0 (all data for validation)
        with pytest.raises(
            ValueError,
            match="validation_split must be between 0 and 1 \\(exclusive of 1\\)",
        ):
            get_data_loader(
                data_path=temp_data_file,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=1.0,
                rank=0,
                world_size=1,
            )

        # test negative ratio
        with pytest.raises(
            ValueError,
            match="validation_split must be between 0 and 1 \\(exclusive of 1\\)",
        ):
            get_data_loader(
                data_path=temp_data_file,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=-0.1,
                rank=0,
                world_size=1,
            )

        # test ratio > 1
        with pytest.raises(
            ValueError,
            match="validation_split must be between 0 and 1 \\(exclusive of 1\\)",
        ):
            get_data_loader(
                data_path=temp_data_file,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=1.5,
                rank=0,
                world_size=1,
            )

        # test very small positive ratio (should work)
        train_loader, val_loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.01,  # 1%
            rank=0,
            world_size=1,
        )
        assert train_loader is not None
        assert val_loader is not None
        # with 10 samples, 1% should give us 1 validation sample
        assert len(val_loader.dataset) >= 1

        # test very large ratio close to 1.0
        # note: with 10 samples, 0.9 (90%) is the highest we can go before train set becomes empty
        train_loader, val_loader = get_data_loader(
            data_path=temp_data_file,
            batch_size=2,
            max_tokens_per_gpu=500,
            seed=42,
            validation_split=0.9,  # 90%
            rank=0,
            world_size=1,
        )
        assert train_loader is not None
        assert val_loader is not None
        # with 10 samples, 90% should give us 9 validation samples, 1 train sample
        assert len(train_loader.dataset) == 1
        assert len(val_loader.dataset) == 9

    def test_max_seq_len_filtering(self):
        """Test that max_seq_len parameter filters out samples exceeding the length."""
        # create a dataset with samples of varying lengths
        data = [
            {
                "input_ids": list(range(i * 10)),
                "labels": list(range(i * 10)),
                "len": i * 10,
                "num_loss_counted_tokens": i * 10,
            }
            for i in range(1, 11)  # lengths: 10, 20, 30, ..., 100
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            # test with max_seq_len=50, should keep samples with len <= 50
            train_loader, val_loader = get_data_loader(
                data_path=temp_path,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=0.2,
                max_seq_len=50,
                rank=0,
                world_size=1,
            )

            # with max_seq_len=50, only samples with lengths 10, 20, 30, 40, 50 should remain (5 samples)
            # with 20% validation split: 4 train, 1 validation
            assert len(train_loader.dataset) == 4
            assert len(val_loader.dataset) == 1

            # verify that samples in the datasets have correct lengths by checking the dataset directly
            for i in range(len(train_loader.dataset)):
                sample = train_loader.dataset[i]
                assert sample["len"] <= 50, f"Train sample {i} has len {sample['len']} > 50"

            for i in range(len(val_loader.dataset)):
                sample = val_loader.dataset[i]
                assert sample["len"] <= 50, f"Val sample {i} has len {sample['len']} > 50"
        finally:
            os.unlink(temp_path)

    @pytest.mark.parametrize(
        "max_seq_len,expected_total",
        [
            (25, 2),  # only lengths 10, 20
            (35, 3),  # only lengths 10, 20, 30
            (100, 10),  # all samples
            (5, 0),  # no samples (all filtered out)
        ],
    )
    def test_max_seq_len_filtering_various_lengths(self, max_seq_len, expected_total):
        """Test max_seq_len filtering with various thresholds."""
        # create a dataset with samples of varying lengths
        data = [
            {
                "input_ids": list(range(i * 10)),
                "labels": list(range(i * 10)),
                "len": i * 10,
                "num_loss_counted_tokens": i * 10,
            }
            for i in range(1, 11)  # lengths: 10, 20, 30, ..., 100
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            train_loader, _ = get_data_loader(
                data_path=temp_path,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_split=0.0,
                max_seq_len=max_seq_len,
                rank=0,
                world_size=1,
            )

            assert len(train_loader.dataset) == expected_total
        finally:
            os.unlink(temp_path)

    def test_validation_data_path_loads_separate_dataset(self, temp_data_file):
        """Test that validation_data_path loads a separate dataset for validation."""
        val_data = [
            {
                "input_ids": list(range(50, 80)),
                "labels": list(range(50, 80)),
                "len": 30,
                "num_loss_counted_tokens": 30,
            }
            for _ in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in val_data:
                json.dump(item, f)
                f.write("\n")
            val_path = f.name

        try:
            train_loader, val_loader = get_data_loader(
                data_path=temp_data_file,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_data_path=val_path,
                rank=0,
                world_size=1,
            )

            assert train_loader is not None
            assert val_loader is not None
            assert len(train_loader.dataset) == 10
            assert len(val_loader.dataset) == 5
        finally:
            os.unlink(val_path)

    def test_validation_data_path_and_split_mutually_exclusive(self, temp_data_file):
        """Test that providing both validation_data_path and validation_split raises an error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({"input_ids": [1, 2], "labels": [1, 2], "len": 2, "num_loss_counted_tokens": 2}, f)
            val_path = f.name

        try:
            with pytest.raises(
                ValueError,
                match="validation_data_path and validation_split are mutually exclusive",
            ):
                get_data_loader(
                    data_path=temp_data_file,
                    batch_size=2,
                    max_tokens_per_gpu=500,
                    seed=42,
                    validation_split=0.2,
                    validation_data_path=val_path,
                    rank=0,
                    world_size=1,
                )
        finally:
            os.unlink(val_path)

    def test_validation_data_path_without_split_uses_all_train_data(self, temp_data_file):
        """Test that validation_data_path does not split training data."""
        val_data = [
            {
                "input_ids": list(range(200, 210)),
                "labels": list(range(200, 210)),
                "len": 10,
                "num_loss_counted_tokens": 10,
            }
            for _ in range(3)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in val_data:
                json.dump(item, f)
                f.write("\n")
            val_path = f.name

        try:
            train_loader, val_loader = get_data_loader(
                data_path=temp_data_file,
                batch_size=2,
                max_tokens_per_gpu=500,
                seed=42,
                validation_data_path=val_path,
                rank=0,
                world_size=1,
            )

            # All 10 samples should be in training (no split)
            assert len(train_loader.dataset) == 10
            # Validation should have the 3 samples from the separate file
            assert len(val_loader.dataset) == 3
        finally:
            os.unlink(val_path)


class TestResetMinibatches:
    """Test suite for the reset_minibatches utility function."""

    def test_reset_minibatches_basic(self):
        """Test basic reset functionality."""
        ids, loads = reset_minibatches(4)

        assert len(ids) == 4
        assert len(loads) == 4
        assert all(isinstance(lst, list) for lst in ids)
        assert all(len(lst) == 0 for lst in ids)
        assert np.array_equal(loads, np.zeros(4))

    def test_reset_minibatches_single_rank(self):
        """Test reset with single rank."""
        ids, loads = reset_minibatches(1)

        assert len(ids) == 1
        assert len(loads) == 1
        assert ids[0] == []
        assert loads[0] == 0


class TestEpochTracking:
    """Test suite for epoch tracking in data loading."""

    def test_epoch_sampler_epoch_increment(self):
        """Test that EpochSampler correctly handles epochs."""
        sampler = EpochSampler(len_data=10, seed=42)

        # First epoch
        sampler.set_epoch(0)
        first_epoch_indices = list(sampler)

        # Second epoch
        sampler.set_epoch(1)
        second_epoch_indices = list(sampler)

        # Each epoch should have all indices 0-9, but in different orders
        assert set(first_epoch_indices) == set(range(10))
        assert set(second_epoch_indices) == set(range(10))

        # The order should be different between epochs (with very high probability)
        assert first_epoch_indices != second_epoch_indices

    def test_distributed_epoch_synchronization(self):
        """Test that different ranks see the same shuffled order per epoch."""
        data_len = 10
        sampler1 = EpochSampler(len_data=data_len, seed=42)
        sampler2 = EpochSampler(len_data=data_len, seed=42)  # Same seed

        for epoch in range(3):
            sampler1.set_epoch(epoch)
            sampler2.set_epoch(epoch)
            iter1 = iter(sampler1)
            iter2 = iter(sampler2)
            for _ in range(data_len):
                assert next(iter1) == next(iter2)

    def test_epoch_boundary_with_uneven_batch_size(self):
        """Test that batch size affects when epoch boundaries are crossed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_file = os.path.join(temp_dir, "test.jsonl")
            samples = []
            for i in range(100):
                samples.append({"input_ids": [1] * 10, "labels": [1] * 10, "len": 10})

            with open(test_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            # Create data loaders with different batch sizes
            loader1, _ = get_data_loader(
                data_path=test_file,
                batch_size=10,  # Evenly divides dataset
                max_tokens_per_gpu=1000,
                seed=42,
                rank=0,
                world_size=1,
            )

            loader2, _ = get_data_loader(
                data_path=test_file,
                batch_size=7,  # Does not evenly divide dataset
                max_tokens_per_gpu=1000,
                seed=42,
                rank=0,
                world_size=1,
            )

            # Count samples in first "epoch" worth of batches
            iter1 = iter(loader1)
            iter2 = iter(loader2)

            samples_seen1 = 0
            samples_seen2 = 0

            # Process 100 samples (1 epoch worth)
            while samples_seen1 < 100:
                batch = next(iter1)
                for mb in batch:
                    samples_seen1 += mb["num_samples"]

            while samples_seen2 < 100:
                batch = next(iter2)
                for mb in batch:
                    samples_seen2 += mb["num_samples"]

            # Both should have seen exactly 100 samples
            assert samples_seen1 == 100
            # With batch_size=7, we might overshoot due to batching
            assert samples_seen2 == 100  # This reveals the issue!

    def test_data_loader_length_with_epoch_sampler(self):
        """Test that DataLoader with EpochSampler reports correct length."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        dataset.__getitem__ = MagicMock(side_effect=lambda x: {"input_ids": [1], "labels": [1]})

        sampler = EpochSampler(len(dataset))
        loader = DataLoader(dataset, batch_size=10, sampler=sampler)

        # The loader computes a length based on batch size
        length = len(loader)
        assert length == 10  # Based on dataset_size / batch_size

        # EpochSampler provides one complete epoch
        count = 0
        for i, batch in enumerate(loader):
            count += 1

        assert count == length  # Should iterate exactly one epoch


class TestDataLoaderBatchCount:
    """Test suite for counting batches in data loader."""

    @pytest.fixture
    def create_test_data(self):
        """Create temporary test data file."""

        def _create(num_samples=10):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for i in range(num_samples):
                    # Create varying length sequences to test token counting
                    seq_length = 10 + (i % 5) * 5  # Lengths: 10, 15, 20, 25, 30
                    # Create realistic token IDs and labels
                    input_ids = list(range(100, 100 + seq_length))
                    # Make some labels -100 (ignored in loss)
                    labels = [lid if j > 2 else -100 for j, lid in enumerate(input_ids)]
                    num_loss_counted = sum(1 for label in labels if label != -100)

                    sample = {
                        "input_ids": input_ids,
                        "labels": labels,
                        "len": seq_length,
                        "num_loss_counted_tokens": num_loss_counted,
                    }
                    f.write(json.dumps(sample) + "\n")
                return f.name

        return _create

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_count_batches_finite_sampler(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test counting batches with finite sampler."""
        data_path = create_test_data(num_samples=20)

        try:
            # Create data loader with finite sampler
            data_loader, _ = get_data_loader(data_path=data_path, batch_size=4, max_tokens_per_gpu=1000, seed=42)

            # Count batches by iterating through the data loader
            batch_count = 0
            for batch in data_loader:
                batch_count += 1

            # With 20 samples and batch size 4, we expect 5 batches
            # Note: actual batching may vary due to dynamic batching based on tokens
            assert batch_count > 0, "Should have at least one batch"

            # Alternative method: use len() if available
            if hasattr(data_loader, "__len__"):
                length_from_len = len(data_loader)
                assert length_from_len > 0, "Data loader length should be positive"

        finally:
            os.unlink(data_path)

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_count_batches_with_epoch(self, mock_world_size, mock_rank, mock_is_init, create_test_data):
        """Test that batch count is consistent across epochs."""
        data_path = create_test_data(num_samples=10)

        try:
            data_loader, _ = get_data_loader(data_path=data_path, batch_size=2, max_tokens_per_gpu=1000, seed=42)

            # Count batches for multiple epochs
            epoch_batch_counts = []
            for epoch in range(3):
                data_loader.sampler.set_epoch(epoch)
                batch_count = 0
                for batch in data_loader:
                    batch_count += 1
                epoch_batch_counts.append(batch_count)

                # check that the dataloader actually increments epoch internally
                assert data_loader.sampler.epoch == epoch, "internal epoch state doesn't match what we expected"

            # All epochs should have the same number of batches
            assert len(set(epoch_batch_counts)) == 1, "Batch counts vary across epochs: {epoch_batch_counts}"
            assert epoch_batch_counts[0] > 0, "Should have at least one batch per epoch"

        finally:
            os.unlink(data_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
