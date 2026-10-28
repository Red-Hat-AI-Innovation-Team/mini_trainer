"""
Integration tests using actual small language models from Transformers.

These tests use tiny model configurations to test real functionality
without requiring large amounts of memory or computation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from mini_trainer.osft_utils import auto_generate_target_osft_config, create_osft_model_class
from mini_trainer.setup_model_for_training import align_model_and_tokenizer, setup_training_components


# TODO: add tests to validate our codebase works with these models
def create_tiny_llama_model():
    """Create a tiny Llama model with ~50k parameters."""
    config = LlamaConfig(
        vocab_size=500,  # Very small vocabulary
        hidden_size=32,  # Tiny hidden size
        intermediate_size=64,  # Small FFN
        num_hidden_layers=2,  # Only 2 layers
        num_attention_heads=2,  # Few attention heads
        num_key_value_heads=1,  # GQA
        max_position_embeddings=64,  # Short sequences
        rope_theta=10000.0,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(config)
    return model, config


def create_tiny_mistral_model():
    """Create a tiny Mistral model with ~50k parameters."""
    config = MistralConfig(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MistralForCausalLM(config)
    return model, config


def create_tiny_qwen2_model():
    """Create a tiny Qwen2 model with ~50k parameters."""
    config = Qwen2Config(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    return model, config


def create_test_data_file(num_samples=10, max_length=50):
    """Create a temporary JSONL file with test data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(num_samples):
            length = min(max_length, 10 + i * 3)
            sample = {
                "input_ids": list(range(length)),
                "labels": list(range(length)),
                "len": length,
                "num_loss_counted_tokens": length - 5,  # Some tokens not counted
            }
            json.dump(sample, f)
            f.write("\n")
        return f.name


class TestModelInitialization:
    """Test model initialization with tiny models.

    TODO: needs more rigorous testing of the setup functions
    """

    def test_model_tokenizer_alignment(self):
        """Test aligning a tiny model with a tokenizer."""
        model, config = create_tiny_llama_model()

        # Set initial config values that differ from tokenizer to test alignment
        model.config.pad_token_id = 999  # Different from tokenizer
        model.config.bos_token_id = 998  # Different from tokenizer
        model.config.eos_token_id = 997  # Different from tokenizer

        # Create a mock tokenizer
        tokenizer = MagicMock()
        tokenizer.__len__ = MagicMock(return_value=500)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        with patch("mini_trainer.setup_model_for_training.log_rank_0"):  # Mock logging
            aligned_model = align_model_and_tokenizer(model, tokenizer)

        # Check alignment happened - model should now match tokenizer
        assert aligned_model.config.pad_token_id == 0
        assert aligned_model.config.bos_token_id == 1
        assert aligned_model.config.eos_token_id == 2

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())),
        reason="No CUDA or XPU accelerator available",
    )
    @patch("mini_trainer.setup_model_for_training.dist.get_rank", return_value=0)
    @patch("mini_trainer.setup_model_for_training.dist.get_world_size", return_value=1)
    @patch("mini_trainer.setup_model_for_training.dist.is_initialized", return_value=False)
    def test_wrap_tiny_model_fsdp(self, mock_dist_init, mock_world_size, mock_rank, monkeypatch):
        """Test FSDP wrapping with a tiny model on the real accelerator (CUDA or XPU)."""
        from mini_trainer import device_utils
        from mini_trainer.setup_model_for_training import wrap_fsdp2

        # Runs on real hardware, so use real device selection rather than the
        # CUDA pin conftest applies to mocked unit tests.
        monkeypatch.delenv("MINI_TRAINER_DEVICE", raising=False)
        model, config = create_tiny_llama_model()
        model = model.to(device_utils.get_device(0))

        # Wrap with FSDP2
        with (
            patch("mini_trainer.setup_model_for_training.init_device_mesh") as mock_mesh,
            patch("mini_trainer.setup_model_for_training.fully_shard") as mock_shard,
        ):
            mock_mesh.return_value = MagicMock()
            mock_shard.side_effect = lambda x, **kwargs: x

            wrapped_model = wrap_fsdp2(model)

            assert wrapped_model is not None
            # Check that sharding was attempted
            # TODO: make sure transformer blocks were also wrapped
            assert mock_shard.called

    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    @patch("mini_trainer.osft_utils.optim_wrapper")
    @patch("transformers.get_scheduler")
    @patch("mini_trainer.setup_model_for_training.wrap_fsdp2")
    def test_training_components_setup_with_tiny_model(self, mock_wrap, mock_sched_fn, mock_opt_wrap, mock_log):
        """Test setting up training components with a tiny model."""
        model, config = create_tiny_llama_model()

        # TODO: ensure proper functions were also called
        mock_wrap.return_value = model
        mock_opt_wrap.side_effect = lambda opt, m: opt

        # Create mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.split_batches = False
        mock_scheduler.step = MagicMock()
        mock_sched_fn.return_value = mock_scheduler

        model, optimizer, scheduler = setup_training_components(
            model, learning_rate=1e-4, num_warmup_steps=10, lr_scheduler="constant"
        )

        assert model is not None
        assert optimizer is not None
        assert scheduler is not None


class TestOSFTModelInitialization:
    """Test OSFT/orthogonal subspace learning with tiny models."""

    def test_osft_model_creation(self):
        """End-to-end OSFT initialization on a tiny GPT2 model."""

        # Create a tiny GPT-2 configuration to keep runtime small
        config = GPT2Config(
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_embd=32,
            n_positions=64,
        )

        # Instantiate a tiny GPT-2 model on CPU
        base_model = GPT2LMHeadModel(config)
        base_model.eval()

        # Dynamically wrap the model with OSFT capabilities
        OSFTModelCls = create_osft_model_class(GPT2LMHeadModel)
        osft_config = auto_generate_target_osft_config(
            base_model,
            model_name_or_class="gpt2",
            target_patterns=["attn.c_proj", "attn.c_attn", "mlp.c_fc", "mlp.c_proj"],
            rank_ratio=0.5,
        )

        osft_model = OSFTModelCls(
            config=config,
            osft_config=osft_config,
            initialize_osft=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32,
            fsdp2_lazy_init=False,
        )

        # initialize OSFT parameters in the non-distributed path
        osft_model.reinitialize_osft(decompose_existing_weights=True)

        # Validate that OSFT structures were created
        assert hasattr(osft_model, "osft_paramspec_registry")
        assert len(osft_model.osft_paramspec_registry) > 0
        assert hasattr(osft_model, "name_mapping")
        assert len(osft_model.name_mapping) > 0

        # ensure we can reconstruct at least one parameter
        first_key = next(iter(osft_model.name_mapping.values()))
        reconstructed = osft_model._reconstruct_weight_by_safe_name(first_key)
        assert isinstance(reconstructed, torch.Tensor)
        assert reconstructed.ndim == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
