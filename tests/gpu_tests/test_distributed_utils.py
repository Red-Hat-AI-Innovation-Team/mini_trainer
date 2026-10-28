"""
Test suite for distributed training utilities.

These tests require GPU/CUDA environment or at least mock NCCL/CUDA components,
so they are placed in the gpu_tests directory.
"""

import os
from unittest.mock import patch

import pytest

from mini_trainer import device_utils
from mini_trainer.utils import check_distributed_is_synchronized, init_distributed_environment


@pytest.mark.gpu
class TestDistributedUtils:
    """Test suite for distributed training utilities."""

    @patch.dict(
        os.environ,
        {"LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0"},
    )
    @patch("mini_trainer.utils.torch.distributed.get_world_size", return_value=1)
    @patch("mini_trainer.utils.torch.distributed.init_process_group")
    @patch("mini_trainer.device_utils.set_device")
    @patch("mini_trainer.utils.check_distributed_is_synchronized")
    @patch("mini_trainer.utils.check_distributed_is_evenly_configured")
    @patch("mini_trainer.utils.torch.distributed.barrier")
    @patch("mini_trainer.utils.log_rank_0")
    def test_init_distributed_environment(
        self,
        mock_log,
        mock_barrier,
        mock_check,
        mock_check_evenly_configured,
        mock_set_device,
        mock_init_pg,
        mock_world_size,
    ):
        """Test distributed environment initialization."""
        init_distributed_environment()

        # Check process group initialization
        mock_init_pg.assert_called_once()
        args, kwargs = mock_init_pg.call_args
        assert args[0] == device_utils.get_dist_backend()
        assert "timeout" in kwargs

        # Check device setting
        mock_set_device.assert_called_once_with(0)

        # Check synchronization check
        mock_check.assert_called_once()

        # Check barrier
        mock_barrier.assert_called_once()

    @patch.dict(os.environ, {"LOCAL_RANK": "0"})
    @patch("mini_trainer.utils.dist.get_rank", return_value=0)
    @patch("mini_trainer.utils.dist.get_world_size", return_value=4)
    @patch("mini_trainer.utils.dist.all_reduce")
    def test_check_distributed_synchronized(self, mock_all_reduce, mock_world_size, mock_rank):
        """Test distributed synchronization check."""

        # Mock successful all_reduce
        def all_reduce_side_effect(tensor, op):
            tensor.fill_(4)  # Simulate 4 processes each adding 1
            return None

        mock_all_reduce.side_effect = all_reduce_side_effect

        # Should not raise
        check_distributed_is_synchronized()

        mock_all_reduce.assert_called_once()

    @patch.dict(os.environ, {"LOCAL_RANK": "0"})
    @patch("mini_trainer.utils.dist.get_rank", return_value=0)
    @patch("mini_trainer.utils.dist.get_world_size", return_value=4)
    @patch("mini_trainer.utils.dist.all_reduce")
    def test_check_distributed_not_synchronized(self, mock_all_reduce, mock_world_size, mock_rank):
        """Test distributed synchronization check failure."""

        # Mock failed all_reduce
        def all_reduce_side_effect(tensor, op):
            tensor.fill_(3)  # Wrong value
            return None

        mock_all_reduce.side_effect = all_reduce_side_effect

        with pytest.raises(AssertionError, match="distributed check failed"):
            check_distributed_is_synchronized()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
