"""
Test suite for model initialization and training setup.

Tests the setup_model, setup_training_components, and related functions
to ensure correct model initialization, FSDP wrapping, and optimizer setup.

TODO: This file needs to be combined with `test_integration_small_models.py`
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest
import torch

from mini_trainer.setup_model_for_training import (
    align_model_and_tokenizer,
    setup_model,
    setup_training_components,
    wrap_fsdp2,
)


class TestAlignModelAndTokenizer:
    """Test suite for model and tokenizer alignment."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.vocab_size = 32000
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.resize_token_embeddings = MagicMock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=32000)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        return tokenizer

    def test_align_matching_vocab_size(self, mock_model, mock_tokenizer):
        """Test alignment when vocab sizes match."""
        result = align_model_and_tokenizer(mock_model, mock_tokenizer)

        assert result == mock_model
        mock_model.resize_token_embeddings.assert_not_called()

    def test_align_resize_vocab(self, mock_model, mock_tokenizer):
        """Test vocab resizing when tokenizer has more tokens."""
        mock_tokenizer.__len__ = MagicMock(return_value=32005)

        with patch("mini_trainer.setup_model_for_training.log_rank_0"):
            mock_model = align_model_and_tokenizer(mock_model, mock_tokenizer)

        # Should resize to next multiple of 8
        mock_model.resize_token_embeddings.assert_called_once_with(32008)

    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    def test_align_fix_special_tokens(self, mock_log, mock_model, mock_tokenizer):
        """Test fixing mismatched special tokens."""
        mock_model.config.pad_token_id = 999
        mock_model.config.bos_token_id = 998
        mock_model.config.eos_token_id = 997

        mock_model = align_model_and_tokenizer(mock_model, mock_tokenizer)

        # Special tokens should be aligned
        assert mock_model.config.pad_token_id == 0
        assert mock_model.config.bos_token_id == 1
        assert mock_model.config.eos_token_id == 2

        # Should have logged warnings
        assert mock_log.call_count == 3

    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    def test_align_none_special_tokens(self, mock_log, mock_model, mock_tokenizer):
        """Test handling of None special tokens."""
        mock_model.config.pad_token_id = None
        mock_tokenizer.pad_token_id = None

        mock_model = align_model_and_tokenizer(mock_model, mock_tokenizer)

        # make sure everything worked properly
        assert mock_model.config.pad_token_id is not None
        assert mock_tokenizer.pad_token_id is not None
        assert mock_model.config.pad_token_id == mock_tokenizer.pad_token_id


class TestWrapFSDP2:
    """Test suite for FSDP2 wrapping."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with transformer layers."""
        model = MagicMock()
        model.device = torch.device("cpu")
        model.config = MagicMock()
        model.config.use_cache = True

        # Mock transformer layers (standard CausalLM structure: model.model.layers)
        layers = [MagicMock() for _ in range(4)]
        model.model = MagicMock(spec=[])  # spec=[] prevents auto-creating attributes
        model.model.layers = layers

        return model

    @patch.dict(os.environ, {"LOCAL_RANK": "0"})
    @patch("mini_trainer.setup_model_for_training.dist.get_rank", return_value=0)
    @patch("mini_trainer.setup_model_for_training.dist.get_world_size", return_value=2)
    @patch("mini_trainer.setup_model_for_training.init_device_mesh")
    @patch("mini_trainer.setup_model_for_training.fully_shard")
    @patch("mini_trainer.setup_model_for_training.ptd_checkpoint_wrapper")
    def test_wrap_fsdp2_basic(
        self,
        mock_checkpoint,
        mock_fully_shard,
        mock_init_mesh,
        mock_world_size,
        mock_rank,
        mock_model,
    ):
        """Test basic FSDP2 wrapping."""
        mock_mesh = MagicMock()
        mock_init_mesh.return_value = mock_mesh
        mock_checkpoint.side_effect = lambda x, **kwargs: x

        wrapped_model = wrap_fsdp2(mock_model)

        # Should disable cache
        assert not mock_model.config.use_cache
        assert wrapped_model == mock_model

        # Should wrap each layer with checkpoint wrapper
        assert mock_checkpoint.call_count == 4

        # Should create device mesh
        mock_init_mesh.assert_called_once_with("cuda", [2], mesh_dim_names=["fsdp"])

        # Should fully shard each layer and the model
        assert mock_fully_shard.call_count == 5  # 4 layers + 1 model

    @patch.dict(os.environ, {"LOCAL_RANK": "1"})
    @patch("mini_trainer.setup_model_for_training.dist.get_rank", return_value=1)
    @patch("mini_trainer.setup_model_for_training.dist.get_world_size", return_value=4)
    @patch("mini_trainer.setup_model_for_training.init_device_mesh")
    @patch("mini_trainer.setup_model_for_training.fully_shard")
    @patch("mini_trainer.setup_model_for_training.ptd_checkpoint_wrapper")
    def test_wrap_fsdp2_multi_gpu(
        self,
        mock_checkpoint,
        mock_fully_shard,
        mock_init_mesh,
        mock_world_size,
        mock_rank,
        mock_model,
    ):
        """Test FSDP2 wrapping with multiple GPUs."""
        mock_mesh = MagicMock()
        mock_init_mesh.return_value = mock_mesh
        mock_checkpoint.side_effect = lambda x, **kwargs: x

        mock_model = wrap_fsdp2(mock_model)

        # Should disable cache
        assert not mock_model.config.use_cache

        # Should wrap each layer with checkpoint wrapper
        assert mock_checkpoint.call_count == 4

        # Should create device mesh with world size 4
        mock_init_mesh.assert_called_once_with("cuda", [4], mesh_dim_names=["fsdp"])

        # Should fully shard each layer and the model (4 layers + 1 model = 5 calls)
        assert mock_fully_shard.call_count == 5

        # Verify that fully_shard was called with the mesh
        for call in mock_fully_shard.call_args_list:
            assert "mesh" in call.kwargs
            assert call.kwargs["mesh"] == mock_mesh

        # FSDP2 handles GPU placement automatically, so no explicit .to() call

    @patch.dict(os.environ, {"LOCAL_RANK": "0"})
    @patch("mini_trainer.setup_model_for_training.dist.get_rank", return_value=0)
    @patch("mini_trainer.setup_model_for_training.dist.get_world_size", return_value=2)
    def test_wrap_fsdp2_no_layers_found(self, mock_world_size, mock_rank, mock_model):
        """
        This test basically verifies that when wrap_fsdp2 cannot find an eligible set of transformer blocks.
        If FSDP2 succeeds in finding a transformer block, you will get errors about torch.distributed
        not being initialized.
        """
        # these are the attributes `wrap_fsdp2` checks for when searching for transformer blocks
        # we disable them so it can't find it
        mock_model.model = None
        mock_model.transformer = None

        with pytest.raises(ValueError, match="Cannot find transformer block container"):
            wrap_fsdp2(mock_model)


class TestSetupModel:
    """Test suite for model setup."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("mini_trainer.setup_model_for_training.align_model_and_tokenizer")
    def test_setup_model_standard(self, mock_align, auto_config, mock_model_cls, mock_tokenizer_cls):
        """Test standard model setup without special features."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LlamaForCausalLM"
        mock_model_cls.return_value = mock_model
        mock_align.return_value = mock_model

        with patch("mini_trainer.setup_model_for_training.log_rank_0"):
            result = setup_model(
                model_name_or_path="meta-llama/Llama-2-7b",
                use_liger_kernels=False,
                osft=False,
                local_rank=0,
            )

        assert result == mock_model
        mock_model_cls.assert_called_once()
        mock_align.assert_called_once_with(mock_model, mock_tokenizer)


class TestSetupTrainingComponents:
    """Test suite for training components setup."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        # Create mock parameters with requires_grad=True
        mock_param = MagicMock()
        mock_param.requires_grad = True
        model.parameters = MagicMock(return_value=[mock_param])
        return model

    @patch("mini_trainer.setup_model_for_training.wrap_fsdp2")
    @patch("transformers.get_scheduler")
    @patch("mini_trainer.setup_model_for_training.torch.optim.AdamW")
    @patch("mini_trainer.osft_utils.optim_wrapper")
    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    def test_setup_training_components_basic(
        self,
        mock_log,
        mock_optim_wrapper,
        mock_adamw,
        mock_scheduler,
        mock_wrap,
        mock_model,
    ):
        """Test basic training components setup."""
        mock_wrapped_model = MagicMock()
        # Set up mock parameters with requires_grad=True for the wrapped model
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_wrapped_model.parameters = MagicMock(return_value=[mock_param])
        mock_wrap.return_value = mock_wrapped_model

        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        mock_wrapped_optimizer = MagicMock()
        mock_optim_wrapper.return_value = mock_wrapped_optimizer

        mock_lr_scheduler = MagicMock()
        mock_lr_scheduler.get_last_lr = MagicMock(return_value=[1e-5])
        mock_scheduler.return_value = mock_lr_scheduler

        model, optimizer, lr_scheduler = setup_training_components(
            mock_model,
            learning_rate=1e-5,
            num_warmup_steps=10,
            lr_scheduler="constant_with_warmup",
        )

        assert model == mock_wrapped_model
        assert optimizer == mock_wrapped_optimizer
        assert lr_scheduler == mock_lr_scheduler

        # Check FSDP2 wrapping
        mock_wrap.assert_called_once_with(mock_model, compile_model=False)

        # Check optimizer creation
        mock_adamw.assert_called_once_with(
            mock_wrapped_model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        # Check optimizer wrapping
        mock_optim_wrapper.assert_called_once_with(mock_optimizer, mock_wrapped_model)

        # Check scheduler creation with new parameters
        mock_scheduler.assert_called_once_with(
            name="constant_with_warmup",
            optimizer=mock_wrapped_optimizer,
            num_warmup_steps=10,
            num_training_steps=None,
            scheduler_specific_kwargs={},
        )

    @patch("mini_trainer.setup_model_for_training.wrap_fsdp2")
    @patch("transformers.get_scheduler")
    @patch("mini_trainer.setup_model_for_training.torch.optim.AdamW")
    @patch("mini_trainer.osft_utils.optim_wrapper")
    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    def test_setup_training_components_different_scheduler(
        self,
        mock_log,
        mock_optim_wrapper,
        mock_adamw,
        mock_scheduler,
        mock_wrap,
        mock_model,
    ):
        """Test setup with different scheduler type."""
        mock_wrap.return_value = mock_model
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        mock_optim_wrapper.return_value = mock_optimizer

        mock_lr_scheduler = MagicMock()
        mock_scheduler.return_value = mock_lr_scheduler

        model, optimizer, lr_scheduler = setup_training_components(
            mock_model, learning_rate=5e-6, num_warmup_steps=100, lr_scheduler="cosine"
        )

        mock_scheduler.assert_called_once_with(
            name="cosine",
            optimizer=mock_optimizer,
            num_warmup_steps=100,
            num_training_steps=None,
            scheduler_specific_kwargs={},
        )


class TestIntegration:
    """Integration tests for model initialization and training setup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("mini_trainer.setup_model_for_training.dist.is_initialized", return_value=False)
    @patch("mini_trainer.setup_model_for_training.AutoTokenizer.from_pretrained")
    def test_model_device_placement(self, mock_tokenizer, mock_dist_init):
        """Test that model is correctly placed on GPU."""
        # This test would require actual model loading
        # Skipping for unit tests to avoid downloading models
        pytest.skip("Integration test requiring actual model download")

    @patch("mini_trainer.osft_utils.optim_wrapper")
    @patch("transformers.get_scheduler")
    @patch("mini_trainer.setup_model_for_training.torch.optim.AdamW")
    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    @patch("mini_trainer.setup_model_for_training.wrap_fsdp2")
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_end_to_end_mock(
        self,
        mock_tok,
        mock_model_cls,
        mock_config,
        mock_wrap,
        mock_log,
        mock_adamw,
        mock_sched,
        mock_opt_wrap,
    ):
        """Test end-to-end flow with mocks."""
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__ = MagicMock(return_value=32000)
        mock_tok.return_value = mock_tokenizer

        # Setup model mock
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LlamaForCausalLM"
        mock_model.config = MagicMock()
        mock_model.config.vocab_size = 32000
        mock_model.parameters = MagicMock(return_value=[MagicMock()])
        mock_model_cls.return_value = mock_model
        mock_wrap.return_value = mock_model

        # Setup optimizer and scheduler mocks
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        mock_opt_wrap.return_value = mock_optimizer
        mock_scheduler = MagicMock()
        mock_sched.return_value = mock_scheduler

        # Setup model
        model = setup_model(
            model_name_or_path="test/model",
            use_liger_kernels=False,
            osft=False,
            local_rank=0,
        )

        # Setup training components
        model, optimizer, scheduler = setup_training_components(
            model, learning_rate=1e-5, num_warmup_steps=10, lr_scheduler="constant"
        )

        assert model is not None
        assert optimizer is not None
        assert scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
