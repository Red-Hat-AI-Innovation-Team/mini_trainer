"""Unit tests for dtype conversion logic in train.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from mini_trainer.train import parse_dtype


class TestDtypeConversion:
    """Test the dtype conversion functionality."""

    def test_parse_dtype_none(self):
        """Test that None input returns None."""
        result = parse_dtype(None)
        assert result is None

    def test_parse_dtype_string_valid(self):
        """Test that valid string dtypes are converted correctly."""
        assert parse_dtype("float16") == torch.float16
        assert parse_dtype("bfloat16") == torch.bfloat16
        assert parse_dtype("float32") == torch.float32
        assert parse_dtype("float64") == torch.float64

    def test_parse_dtype_string_invalid(self):
        """Test that invalid string dtypes raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype string: 'float8'"):
            parse_dtype("float8")

        with pytest.raises(ValueError, match="Unsupported dtype string: 'invalid'"):
            parse_dtype("invalid")

        with pytest.raises(ValueError, match="Unsupported dtype string: 'int32'"):
            parse_dtype("int32")

    def test_parse_dtype_torch_dtype(self):
        """Test that torch.dtype objects are passed through unchanged."""
        assert parse_dtype(torch.float16) == torch.float16
        assert parse_dtype(torch.bfloat16) == torch.bfloat16
        assert parse_dtype(torch.float32) == torch.float32
        assert parse_dtype(torch.float64) == torch.float64

    def test_parse_dtype_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Invalid dtype type"):
            parse_dtype(123)  # integer

        with pytest.raises(TypeError, match="Invalid dtype type"):
            parse_dtype(12.5)  # float

        with pytest.raises(TypeError, match="Invalid dtype type"):
            parse_dtype([])  # list


class TestMainFunctionDtypeIntegration:
    """Integration tests for dtype parameters in main function."""

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.train.get_node_rank", return_value=0)
    @patch("mini_trainer.train.setup_model")
    @patch("mini_trainer.train.setup_training_components")
    @patch("mini_trainer.train.train")
    @patch("mini_trainer.train.get_data_loader")
    @patch("mini_trainer.train.calculate_num_training_steps")
    @patch("mini_trainer.train.destroy_distributed_environment")
    @patch("mini_trainer.train.init_distributed_environment")
    @patch("torch.distributed.get_rank", return_value=0)
    def test_main_dtype_defaults(
        self,
        mock_get_rank,
        mock_init_dist,
        mock_destroy_dist,
        mock_calc_steps,
        mock_data_loader,
        mock_train_fn,
        mock_setup_components,
        mock_setup_model,
        mock_get_node_rank,
        mock_world_size,
    ):
        """Test main function with default dtype values."""
        # Setup mocks
        mock_calc_steps.return_value = 1000
        mock_data_loader.return_value = (MagicMock(), MagicMock())
        mock_setup_model.return_value = MagicMock()
        mock_setup_components.return_value = (MagicMock(), MagicMock(), MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Import main function from train module
            from mini_trainer.train import main

            # Call main with defaults (should not raise any errors)
            try:
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    # Using defaults: osft_upcast_dtype="float32"
                )
            except SystemExit:
                pass  # typer might call sys.exit, which is fine for this test

            # Verify setup_model was called with converted dtypes
            mock_setup_model.assert_called_once()
            call_kwargs = mock_setup_model.call_args.kwargs

            # Check that the torch dtypes were passed correctly
            assert call_kwargs["osft_upcast_dtype"] == torch.float32

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.train.get_node_rank", return_value=0)
    @patch("mini_trainer.train.setup_model")
    @patch("mini_trainer.train.setup_training_components")
    @patch("mini_trainer.train.train")
    @patch("mini_trainer.train.get_data_loader")
    @patch("mini_trainer.train.calculate_num_training_steps")
    @patch("mini_trainer.train.destroy_distributed_environment")
    @patch("mini_trainer.train.init_distributed_environment")
    @patch("torch.distributed.get_rank")
    def test_main_dtype_custom_strings(
        self,
        mock_get_rank,
        mock_init_dist,
        mock_destroy_dist,
        mock_calc_steps,
        mock_data_loader,
        mock_train_fn,
        mock_setup_components,
        mock_setup_model,
        mock_get_node_rank,
        mock_world_size,
    ):
        """Test main function with custom string dtype values."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_calc_steps.return_value = 1000
        mock_data_loader.return_value = (MagicMock(), MagicMock())
        mock_setup_model.return_value = MagicMock()
        mock_setup_components.return_value = (MagicMock(), MagicMock(), MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            from mini_trainer.train import main

            try:
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft_upcast_dtype="bfloat16",
                )
            except SystemExit:
                pass

            # Verify setup_model was called with converted dtypes
            mock_setup_model.assert_called_once()
            call_kwargs = mock_setup_model.call_args.kwargs

            assert call_kwargs["osft_upcast_dtype"] == torch.bfloat16

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.train.get_node_rank", return_value=0)
    @patch("mini_trainer.train.setup_model")
    @patch("mini_trainer.train.setup_training_components")
    @patch("mini_trainer.train.train")
    @patch("mini_trainer.train.get_data_loader")
    @patch("mini_trainer.train.calculate_num_training_steps")
    @patch("mini_trainer.train.destroy_distributed_environment")
    @patch("mini_trainer.train.init_distributed_environment")
    @patch("torch.distributed.get_rank")
    def test_main_dtype_none_values(
        self,
        mock_get_rank,
        mock_init_dist,
        mock_destroy_dist,
        mock_calc_steps,
        mock_data_loader,
        mock_train_fn,
        mock_setup_components,
        mock_setup_model,
        mock_get_node_rank,
        mock_world_size,
    ):
        """Test main function with None dtype values."""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_calc_steps.return_value = 1000
        mock_data_loader.return_value = (MagicMock(), MagicMock())
        mock_setup_model.return_value = MagicMock()
        mock_setup_components.return_value = (MagicMock(), MagicMock(), MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            from mini_trainer.train import main

            try:
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft_upcast_dtype=None,
                )
            except SystemExit:
                pass

            # Verify setup_model was called with None values
            mock_setup_model.assert_called_once()
            call_kwargs = mock_setup_model.call_args.kwargs

            assert call_kwargs["osft_upcast_dtype"] is None

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.train.get_node_rank", return_value=0)
    @patch("mini_trainer.train.init_distributed_environment")
    @patch("torch.distributed.get_rank")
    def test_main_invalid_dtype_raises_error(self, mock_get_rank, mock_init_dist, mock_get_node_rank, mock_world_size):
        """Test that main function raises error for invalid dtype strings."""
        mock_get_rank.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            from mini_trainer.train import main

            with pytest.raises(ValueError, match="Unsupported dtype string: 'invalid_dtype'"):
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft_upcast_dtype="invalid_dtype",
                )


class TestDtypeParameterLogging:
    """Test that dtype parameters are properly logged."""

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.train.get_node_rank", return_value=0)
    @patch("mini_trainer.train.setup_model")
    @patch("mini_trainer.train.setup_training_components")
    @patch("mini_trainer.train.train")
    @patch("mini_trainer.train.get_data_loader")
    @patch("mini_trainer.train.calculate_num_training_steps")
    @patch("mini_trainer.train.destroy_distributed_environment")
    @patch("mini_trainer.train.init_distributed_environment")
    @patch("torch.distributed.get_rank")
    def test_dtype_parameters_logged(
        self,
        mock_get_rank,
        mock_init_dist,
        mock_destroy_dist,
        mock_calc_steps,
        mock_data_loader,
        mock_train_fn,
        mock_setup_components,
        mock_setup_model,
        mock_get_node_rank,
        mock_world_size,
    ):
        """Test that dtype parameters are included in logged parameters."""
        # Setup mocks
        mock_get_rank.return_value = 0  # Ensure we're on rank 0 for logging
        mock_calc_steps.return_value = 1000
        mock_data_loader.return_value = (MagicMock(), MagicMock())
        mock_setup_model.return_value = MagicMock()
        mock_setup_components.return_value = (MagicMock(), MagicMock(), MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            from mini_trainer.train import main

            try:
                main(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=1,
                    max_tokens_per_gpu=100,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft_upcast_dtype="bfloat16",
                )
            except SystemExit:
                pass

            # Check that parameters were logged to file
            params_file = Path(tmpdir) / "training_params.json"
            assert params_file.exists()

            import json

            with open(params_file) as f:
                logged_params = json.load(f)

            # Verify dtype parameters are in the logged parameters
            assert "osft_upcast_dtype" in logged_params
            assert logged_params["osft_upcast_dtype"] == "bfloat16"
            assert "osft_output_dtype" not in logged_params
