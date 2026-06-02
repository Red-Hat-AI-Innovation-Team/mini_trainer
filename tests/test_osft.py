"""
Comprehensive tests for OSFT (Orthogonal Subspace Fine-Tuning) and SVD functionality.

Tests validate:
1. osft_unfreeze_rank_ratio validation in API
2. osft_target_patterns passing through API
3. SVD config generation with custom patterns
4. Integration with setup_model
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import mini_trainer.osft_utils as osft_module
from mini_trainer.api_train import run_training
from mini_trainer.osft_utils import (
    MODEL_CONFIGS,
    _get_model_patterns_from_name,
    _load_model_memory_efficient,
    auto_generate_target_osft_config,
    create_osft_model_class,
    get_model_config,
    is_osft_param,
    optim_wrapper,
    project_gradient_to_orthogonal_space,
    project_parameter_to_orthogonal_space,
)
from mini_trainer.setup_model_for_training import setup_model
from mini_trainer.training_types import TorchrunArgs, TrainingArgs
from tests.test_utils.orthogonality import (
    OrthogonalityTracker,
    check_gradient_orthogonality,
    check_parameter_orthogonality,
    compute_angle_differences,
)


class TestOSFTAPIValidation:
    """Test OSFT parameter validation in the API."""

    @patch("mini_trainer.api_train.StreamablePopen")
    def test_osft_requires_rank_ratio(self, mock_popen_class):
        """Test that osft=True requires osft_unfreeze_rank_ratio to be provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            # osft=True but osft_unfreeze_rank_ratio=None should raise error
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=None,  # This should cause an error
            )

            mock_popen = MagicMock()
            # it should not even run this, so return value doesn't matter here
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen

            with pytest.raises(
                ValueError,
                match="osft_unfreeze_rank_ratio is required when osft is True",
            ):
                run_training(torch_args, train_args)

            # shouldnt have even gotten run
            assert mock_popen_class.call_count == 0

    @patch("mini_trainer.api_train.StreamablePopen")
    def test_osft_with_valid_rank_ratio(self, mock_popen_class):
        """Test that osft=True with valid unfreeze_rank_ratio passes validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,  # Valid ratio
            )

            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0  # Success
            mock_popen_class.return_value = mock_popen

            run_training(torch_args, train_args)

            # Verify command includes osft parameters
            call_args = mock_popen_class.call_args
            _, command = call_args[0]

            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.5" in command
            assert mock_popen_class.call_count > 0

    def test_osft_unfreeze_rank_ratio_not_required_when_osft_false(self):
        """Test that osft_unfreeze_rank_ratio is not required when osft=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=False,
                osft_unfreeze_rank_ratio=None,  # This should be fine
            )

            with patch("mini_trainer.api_train.StreamablePopen") as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen

                # Should not raise error
                run_training(torch_args, train_args)

                # Verify osft parameters not in command
                call_args = mock_popen_class.call_args
                _, command = call_args[0]

                assert "--osft" not in command
                assert all(not arg.startswith("--osft-unfreeze-rank-ratio") for arg in command)
                assert mock_popen_class.call_count > 0

    @patch("mini_trainer.api_train.StreamablePopen")
    def test_osft_target_patterns_passed_through(self, mock_popen_class):
        """Test that osft_target_patterns are correctly passed through the API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            test_patterns = ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.75,
                osft_target_patterns=test_patterns,
            )

            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen

            run_training(torch_args, train_args)

            # Verify command includes target patterns
            call_args = mock_popen_class.call_args
            _, command = call_args[0]

            assert "--osft" in command
            assert "--osft-unfreeze-rank-ratio=0.75" in command
            # Find the target patterns argument
            patterns_arg = None
            for arg in command:
                if arg.startswith("--osft-target-patterns="):
                    patterns_arg = arg
                    break

            assert patterns_arg is not None
            # The patterns should be passed as a list string
            expected = "--osft-target-patterns=self_attn.q_proj,self_attn.k_proj,mlp.gate_proj"
            assert patterns_arg == expected

    def test_osft_target_patterns_empty_list(self):
        """Test that empty osft_target_patterns list is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,
                osft_target_patterns=[],  # Empty list
            )

            with patch("mini_trainer.api_train.StreamablePopen") as mock_popen_class:
                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen

                run_training(torch_args, train_args)

                call_args = mock_popen_class.call_args
                _, command = call_args[0]

                # Empty list is treated same as None - not passed
                # This is reasonable as empty list means no custom patterns
                patterns_arg = None
                for arg in command:
                    if arg.startswith("--osft-target-patterns="):
                        patterns_arg = arg
                        break

                assert patterns_arg is None

    @patch("mini_trainer.api_train.StreamablePopen")
    def test_osft_target_patterns_none_not_passed(self, mock_popen_class):
        """Test that None osft_target_patterns is not passed to command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_args = TorchrunArgs(nproc_per_node=8)
            train_args = TrainingArgs(
                model_name_or_path="test-model",
                data_path="test.jsonl",
                batch_size=32,
                max_tokens_per_gpu=1000,
                learning_rate=1e-5,
                output_dir=tmpdir,
                osft=True,
                osft_unfreeze_rank_ratio=0.5,
                osft_target_patterns=None,  # None should not be passed
            )

            mock_popen = MagicMock()
            mock_popen.poll.return_value = 0
            mock_popen_class.return_value = mock_popen

            run_training(torch_args, train_args)

            call_args = mock_popen_class.call_args
            _, command = call_args[0]

            # None should result in no target patterns argument
            assert all(not arg.startswith("--osft-target-patterns") for arg in command)

    @patch("mini_trainer.api_train.StreamablePopen")
    def test_various_rank_ratios(self, mock_popen_class):
        """Test that different rank ratios are correctly passed."""
        rank_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for ratio in rank_ratios:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch_args = TorchrunArgs(nproc_per_node=8)
                train_args = TrainingArgs(
                    model_name_or_path="test-model",
                    data_path="test.jsonl",
                    batch_size=32,
                    max_tokens_per_gpu=1000,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    osft=True,
                    osft_unfreeze_rank_ratio=ratio,
                )

                mock_popen = MagicMock()
                mock_popen.poll.return_value = 0
                mock_popen_class.return_value = mock_popen

                run_training(torch_args, train_args)

                call_args = mock_popen_class.call_args
                _, command = call_args[0]

                assert f"--osft-unfreeze-rank-ratio={ratio}" in command


class TestOSFTConfigGeneration:
    """Test SVD configuration generation with custom patterns."""

    def test_get_model_patterns_from_name(self):
        """Test pattern detection from model names."""
        # Test known model types
        # We need to make sure that these models are tested:
        # - Llama
        # - Qwen
        # - Mistral
        # - Phi-4
        assert _get_model_patterns_from_name("llama") == MODEL_CONFIGS["llama"]["patterns"]
        assert _get_model_patterns_from_name("gpt-j-6b") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("gptj") == MODEL_CONFIGS["gpt-j"]["patterns"]
        assert _get_model_patterns_from_name("opt-350m") == MODEL_CONFIGS["opt"]["patterns"]
        assert _get_model_patterns_from_name("qwen2-7b") == MODEL_CONFIGS["qwen"]["patterns"]
        assert _get_model_patterns_from_name("gemma-2b") == MODEL_CONFIGS["gemma"]["patterns"]
        assert _get_model_patterns_from_name("mistral") == MODEL_CONFIGS["mistral"]["patterns"]
        assert _get_model_patterns_from_name("mistral-7b") == MODEL_CONFIGS["mistral"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-4") == MODEL_CONFIGS["phi3"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-3") == MODEL_CONFIGS["phi3"]["patterns"]
        assert _get_model_patterns_from_name("microsoft/Phi-4-mini-instruct") == MODEL_CONFIGS["phi3"]["patterns"]

        # Test default fallback
        assert _get_model_patterns_from_name("unknown-model") == MODEL_CONFIGS["default"]["patterns"]

    def test_get_model_config_with_custom_patterns(self):
        """Test that custom patterns override model defaults."""
        custom_patterns = ["custom.layer1", "custom.layer2"]

        # Custom patterns should override model-specific patterns
        patterns = get_model_config("llama", target_patterns=custom_patterns)
        assert patterns == custom_patterns

        # Custom patterns should override default
        patterns = get_model_config(None, target_patterns=custom_patterns)
        assert patterns == custom_patterns

    def test_get_model_config_without_custom_patterns(self):
        """Test model config retrieval without custom patterns."""
        # Should get model-specific patterns
        patterns = get_model_config("llama", target_patterns=None)
        assert patterns == MODEL_CONFIGS["llama"]["patterns"]

        # Should get default patterns
        patterns = get_model_config(None, target_patterns=None)
        assert patterns == MODEL_CONFIGS["default"]["patterns"]

    def test_auto_generate_osft_config_with_custom_patterns(self):
        """Test OSFT config generation with custom target patterns."""
        # Create a mock model with various layers
        mock_model = MagicMock()
        mock_params = [
            ("layer1.self_attn.q_proj.weight", torch.zeros(128, 64)),
            ("layer1.self_attn.k_proj.weight", torch.zeros(128, 64)),
            ("layer1.mlp.gate_proj.weight", torch.zeros(256, 128)),
            ("layer2.custom_proj.weight", torch.zeros(100, 50)),
            ("layer2.another_proj.weight", torch.zeros(200, 100)),
        ]
        mock_model.named_parameters.return_value = mock_params

        # Test with custom patterns
        custom_patterns = ["custom_proj", "another_proj"]
        config = auto_generate_target_osft_config(mock_model, target_patterns=custom_patterns, rank_ratio=0.5)

        # Should only include layers matching custom patterns
        assert "layer2.custom_proj.weight" in config
        assert "layer2.another_proj.weight" in config
        assert "layer1.self_attn.q_proj.weight" not in config
        assert "layer1.self_attn.k_proj.weight" not in config
        assert "layer1.mlp.gate_proj.weight" not in config

        # Check rank values
        assert config["layer2.custom_proj.weight"] == 25  # min(100, 50) * 0.5
        assert config["layer2.another_proj.weight"] == 50  # min(200, 100) * 0.5

    def test_auto_generate_osft_config_with_rank_ratio(self):
        """Test that rank_ratio correctly affects the generated config."""
        mock_model = MagicMock()
        mock_params = [
            ("layer.proj.weight", torch.zeros(100, 80)),
        ]
        mock_model.named_parameters.return_value = mock_params

        # Test different rank ratios
        for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
            config = auto_generate_target_osft_config(mock_model, target_patterns=["proj"], rank_ratio=ratio)

            expected_rank = int(80 * ratio)  # min(100, 80) * ratio
            assert config["layer.proj.weight"] == expected_rank

    def test_auto_generate_svd_config_edge_cases(self):
        """Test edge cases in SVD config generation."""
        mock_model = MagicMock()

        # Test with rank_ratio >= 1.0 (should cap at full_rank - 1)
        mock_params = [("layer.proj.weight", torch.zeros(50, 50))]
        mock_model.named_parameters.return_value = mock_params

        config = auto_generate_target_osft_config(mock_model, target_patterns=["proj"], rank_ratio=1.0)
        assert config["layer.proj.weight"] == 49  # full_rank - 1

        # Test with 1D parameters (should be skipped)
        mock_params = [
            ("layer.bias", torch.zeros(100)),  # 1D parameter
            ("layer.weight", torch.zeros(100, 50)),  # 2D parameter
        ]
        mock_model.named_parameters.return_value = mock_params

        config = auto_generate_target_osft_config(mock_model, target_patterns=["layer"], rank_ratio=0.5)

        assert "layer.bias" not in config  # 1D should be skipped
        assert "layer.weight" in config  # 2D should be included

    def test_is_osft_param_function(self):
        """Test the is_osft_param utility function."""
        osft_config = {
            "layer1.weight": 10,
            "layer2.weight": 0,  # 0 means not OSFT
        }

        # 2D param with positive rank in config
        param_2d = torch.zeros(100, 50)
        assert is_osft_param("layer1.weight", param_2d, osft_config) is True

        # 2D param with 0 rank in config
        assert is_osft_param("layer2.weight", param_2d, osft_config) is False

        # 2D param not in config
        assert is_osft_param("layer3.weight", param_2d, osft_config) is False

        # 1D param (should be False regardless)
        param_1d = torch.zeros(100)
        assert is_osft_param("layer1.weight", param_1d, osft_config) is False

    def _create_tiny_llama_model(self):
        """Create a tiny Llama model for testing."""
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            pytest.skip("LlamaForCausalLM not available in this transformers version")

        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            rope_theta=10000.0,
        )
        model = LlamaForCausalLM(config)
        return model, config, "llama"

    def _create_tiny_mistral_model(self):
        """Create a tiny Mistral model for testing."""
        try:
            from transformers import MistralConfig, MistralForCausalLM
        except ImportError:
            pytest.skip("MistralForCausalLM not available in this transformers version")

        config = MistralConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            sliding_window=8,
        )
        model = MistralForCausalLM(config)
        return model, config, "mistral"

    def _create_tiny_qwen2_model(self):
        """Create a tiny Qwen2 model for testing."""
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            pytest.skip("Qwen2ForCausalLM not available in this transformers version")

        config = Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
        model = Qwen2ForCausalLM(config)
        return model, config, "qwen"

    def _create_tiny_phi4_model(self):
        """Create a tiny Phi-4 model for testing."""
        try:
            from transformers import Phi3Config, Phi3ForCausalLM
        except ImportError:
            pytest.skip("Phi3ForCausalLM not available in this transformers version")

        config = Phi3Config(
            vocab_size=1000,  # Large enough for pad_token_id
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=999,  # Set within vocab_size
            eos_token_id=998,
            bos_token_id=997,
        )
        model = Phi3ForCausalLM(config)
        return model, config, "phi3"

    def _create_tiny_gptj_model(self):
        """Create a tiny GPT-J model for testing."""
        try:
            from transformers import GPTJConfig, GPTJForCausalLM
        except ImportError:
            pytest.skip("GPTJForCausalLM not available in this transformers version")

        config = GPTJConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            rotary_dim=8,  # Required for GPT-J
            max_position_embeddings=16,
        )
        model = GPTJForCausalLM(config)
        return model, config, "gpt-j"

    def _create_tiny_gptneo_model(self):
        """Create a tiny GPT-NEO model for testing."""
        try:
            from transformers import GPTNeoConfig, GPTNeoForCausalLM
        except ImportError:
            pytest.skip("GPTNeoForCausalLM not available in this transformers version")

        config = GPTNeoConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
            attention_types=[["global", 2], ["global", 2]],  # Required format
            attention_layers=["global", "global"],
        )
        model = GPTNeoForCausalLM(config)
        return model, config, "gpt-neo"

    def _create_tiny_opt_model(self):
        """Create a tiny OPT model for testing."""
        try:
            from transformers import OPTConfig, OPTForCausalLM
        except ImportError:
            pytest.skip("OPTForCausalLM not available in this transformers version")

        config = OPTConfig(
            vocab_size=100,
            hidden_size=16,
            ffn_dim=32,  # OPT uses ffn_dim instead of intermediate_size
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
        )
        model = OPTForCausalLM(config)
        return model, config, "opt"

    def _create_tiny_gemma_model(self):
        """Create a tiny GEMMA model for testing."""
        try:
            from transformers import GemmaConfig, GemmaForCausalLM
        except ImportError:
            pytest.skip("GemmaForCausalLM not available in this transformers version")

        config = GemmaConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=0,
        )
        model = GemmaForCausalLM(config)
        return model, config, "gemma"

    def _create_tiny_granite_model(self):
        """Create a tiny GRANITE model for testing."""
        try:
            from transformers import GraniteConfig, GraniteForCausalLM
        except ImportError:
            pytest.skip("GraniteForCausalLM not available in this transformers version")

        config = GraniteConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            pad_token_id=0,
        )
        model = GraniteForCausalLM(config)
        return model, config, "granite"

    def _get_model_layer_patterns(self, model_type, config, layer_idx):
        """Get expected layer patterns for different model types."""
        if model_type in ["llama", "mistral", "qwen", "gemma", "granite"]:
            # These models use the same layer structure: model.layers.{idx}
            layer_prefix = f"model.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
                f"{layer_prefix}.self_attn.o_proj.weight",
                f"{layer_prefix}.mlp.gate_proj.weight",
                f"{layer_prefix}.mlp.up_proj.weight",
                f"{layer_prefix}.mlp.down_proj.weight",
            ]
        elif model_type == "phi3":
            # Phi-3/Phi-4 models use combined projections: model.layers.{idx}
            layer_prefix = f"model.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.qkv_proj.weight",  # Combined q/k/v projection
                f"{layer_prefix}.self_attn.o_proj.weight",  # Output projection
                f"{layer_prefix}.mlp.gate_up_proj.weight",  # Combined gate/up projection
                f"{layer_prefix}.mlp.down_proj.weight",  # Down projection
            ]
        elif model_type == "gpt-j":
            # GPT-J uses h.{idx} instead of layers.{idx}
            layer_prefix = f"transformer.h.{layer_idx}"
            return [
                f"{layer_prefix}.attn.q_proj.weight",
                f"{layer_prefix}.attn.k_proj.weight",
                f"{layer_prefix}.attn.v_proj.weight",
                f"{layer_prefix}.attn.out_proj.weight",
                f"{layer_prefix}.mlp.fc_in.weight",
                f"{layer_prefix}.mlp.fc_out.weight",
            ]
        elif model_type == "gpt-neo":
            # GPT-NEO uses h.{idx} with nested attention structure
            layer_prefix = f"transformer.h.{layer_idx}"
            return [
                f"{layer_prefix}.attn.attention.q_proj.weight",
                f"{layer_prefix}.attn.attention.k_proj.weight",
                f"{layer_prefix}.attn.attention.v_proj.weight",
                f"{layer_prefix}.attn.attention.out_proj.weight",
                f"{layer_prefix}.mlp.c_fc.weight",
                f"{layer_prefix}.mlp.c_proj.weight",
            ]
        elif model_type == "opt":
            # OPT uses decoder.layers.{idx}
            layer_prefix = f"model.decoder.layers.{layer_idx}"
            return [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
                f"{layer_prefix}.self_attn.out_proj.weight",
                f"{layer_prefix}.fc1.weight",
                f"{layer_prefix}.fc2.weight",
            ]
        else:
            # For future model types, we can add specific handling
            raise NotImplementedError(f"Layer patterns not implemented for {model_type}")

    @pytest.mark.parametrize(
        "model_creator",
        [
            "_create_tiny_llama_model",
            "_create_tiny_mistral_model",
            "_create_tiny_qwen2_model",
            "_create_tiny_phi4_model",
            "_create_tiny_gptj_model",
            "_create_tiny_gptneo_model",
            "_create_tiny_opt_model",
            "_create_tiny_gemma_model",
            "_create_tiny_granite_model",
        ],
    )
    def test_model_state_dict_pattern_matching(self, model_creator):
        """Test that model state dicts correctly match expected OSFT patterns."""
        # Get the model creator method and create the model
        creator_method = getattr(self, model_creator)
        model, config, model_type = creator_method()

        # Get the OSFT config using the model
        osft_config = auto_generate_target_osft_config(model, model_name_or_class=model_type, rank_ratio=0.5)

        # Expected patterns from MODEL_CONFIGS
        expected_patterns = MODEL_CONFIGS[model_type]["patterns"]

        # Verify that all expected patterns are found in the model's state dict
        model_param_names = [name for name, _ in model.named_parameters()]

        # Check that each expected pattern matches at least one parameter
        for pattern in expected_patterns:
            matching_params = [name for name in osft_config.keys() if pattern in name]
            assert len(matching_params) > 0, f"Pattern '{pattern}' not found in OSFT config for {model_type}"

            # Also verify these parameters exist in the actual model
            model_matches = [name for name in model_param_names if pattern in name and ".weight" in name]
            assert len(model_matches) > 0, f"Pattern '{pattern}' not found in model parameters for {model_type}"

        # Verify that the OSFT config only contains parameters matching our patterns
        for param_name in osft_config.keys():
            assert any(pattern in param_name for pattern in expected_patterns), (
                f"Parameter '{param_name}' doesn't match any expected pattern for {model_type}"
            )

        # Verify correct number of layers are matched (2 layers as configured)
        for i in range(config.num_hidden_layers):
            expected_layer_params = self._get_model_layer_patterns(model_type, config, i)
            for expected_param in expected_layer_params:
                assert expected_param in osft_config, (
                    f"Expected parameter '{expected_param}' not found in OSFT config for {model_type}"
                )

        # Verify rank values are correctly calculated
        for param_name, rank in osft_config.items():
            param = dict(model.named_parameters())[param_name]
            expected_rank = int(min(param.shape) * 0.5)
            if expected_rank >= min(param.shape):
                expected_rank = min(param.shape) - 1
            assert rank == expected_rank, (
                f"Rank mismatch for {param_name} in {model_type}: got {rank}, expected {expected_rank}"
            )


class TestOSFTModelCreation:
    """Test OSFT model class creation and initialization."""

    def test_create_osft_model_class(self):
        """Test that create_osft_model_class creates a valid subclass."""

        # Create a simple mock base class
        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(10, 10)

        # Create OSFT model class
        OSFTModelClass = create_osft_model_class(MockModel)

        # Check class inheritance
        assert issubclass(OSFTModelClass, MockModel)
        assert OSFTModelClass.__name__ == "MockModelWithOSFT"

        # Check that required methods exist
        assert hasattr(OSFTModelClass, "reinitialize_osft")
        assert hasattr(OSFTModelClass, "reinitialize_osft_distributed")
        assert hasattr(OSFTModelClass, "project_gradients")
        assert hasattr(OSFTModelClass, "from_pretrained")

    def test_osft_model_initialization_without_osft(self):
        """Test OSFT model can be initialized without OSFT decomposition."""

        class MockModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.dtype = torch.float32

        OSFTModelClass = create_osft_model_class(MockModel)

        # Initialize without OSFT
        config = MagicMock()
        model = OSFTModelClass(config, osft_config={}, initialize_osft=False)

        assert model.osft_config == {}
        assert hasattr(model, "osft_params")
        assert len(model.osft_params) == 0


class TestSetupModelIntegration:
    """Test integration of OSFT options with setup_model function."""

    @patch("mini_trainer.setup_model_for_training.log_rank_0")
    @patch("transformers.AutoConfig")
    @patch("mini_trainer.setup_model_for_training.get_model_class_from_config")
    @patch("transformers.AutoModelForCausalLM")
    @patch("mini_trainer.setup_model_for_training.AutoTokenizer")
    @patch("mini_trainer.setup_model_for_training.AutoConfig")
    @patch("mini_trainer.osft_utils.auto_generate_target_osft_config")
    @patch("mini_trainer.setup_model_for_training.create_osft_model_class")
    def test_osft_params_flow_through_setup(
        self,
        mock_osft_class,
        mock_auto_config,
        mock_setup_auto_config,
        mock_tokenizer_cls,
        mock_model_cls,
        mock_get_model_class,
        mock_transformers_auto_config,
        mock_log,
    ):
        """Test that OSFT parameters flow through the setup correctly."""
        # Test that OSFT model creation gets the right parameters
        mock_auto_config.return_value = {"layer.weight": 10}

        # Create mock OSFT instance
        mock_osft_instance = MagicMock()
        mock_osft_instance.config = MagicMock()
        mock_osft_instance.config.vocab_size = 1000
        mock_osft_instance.dtype = torch.float32
        mock_osft_instance.reinitialize_osft = MagicMock()
        mock_osft_instance.named_parameters = MagicMock(return_value=[])
        mock_osft_instance.parameters = MagicMock(return_value=[])

        # Create a function that builds the OSFT class
        def create_mock_osft_class(base_cls):
            class MockOSFTModelCls(base_cls):
                last_kwargs = {}  # Store kwargs for verification

                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    # Store the kwargs for verification
                    cls.last_kwargs = kwargs
                    # Set attributes on the instance
                    mock_osft_instance.upcast_dtype = kwargs.get("upcast_dtype", torch.float32)
                    if "output_dtype" in kwargs and kwargs["output_dtype"] is not None:
                        mock_osft_instance.output_dtype = kwargs["output_dtype"]
                    return mock_osft_instance

            # Store the class for later verification
            create_mock_osft_class.osft_class = MockOSFTModelCls
            return MockOSFTModelCls

        mock_osft_class.side_effect = create_mock_osft_class

        # Mock tokenizer and base model
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Create a proper base model class
        class MockBaseModelClass:
            __name__ = "LlamaForCausalLM"

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                # This is what super().from_pretrained() will call
                return mock_osft_instance

        mock_base_model = MagicMock()
        mock_base_model.config = MagicMock()
        mock_base_model.config.vocab_size = 1000
        mock_base_model.__class__ = MockBaseModelClass
        mock_model_cls.from_pretrained.return_value = mock_base_model

        # Mock get_model_class_from_config to return the base model class
        mock_get_model_class.return_value = MockBaseModelClass

        # Mock AutoConfig globally (used in osft_utils) to return a non-GPT-OSS config
        mock_osft_config = MagicMock()
        mock_osft_config.model_type = "llama"  # Not GPT-OSS
        mock_transformers_auto_config.from_pretrained.return_value = mock_osft_config

        # Call setup_model with OSFT params
        setup_model(
            osft=True,
            local_rank=0,
            osft_rank_ratio=0.75,
            osft_target_patterns=["custom.layer1", "custom.layer2"],
            model_name_or_path="test-model",
        )

        # Verify the OSFT model class was created
        mock_osft_class.assert_called_once()

        # Verify from_pretrained was called with the right params
        # Get the OSFT class that was created
        osft_cls = create_mock_osft_class.osft_class
        assert "rank_ratio" in osft_cls.last_kwargs
        assert osft_cls.last_kwargs["rank_ratio"] == 0.75
        assert "target_patterns" in osft_cls.last_kwargs
        assert osft_cls.last_kwargs["target_patterns"] == [
            "custom.layer1",
            "custom.layer2",
        ]


class TestEndToEndOSFT:
    """End-to-end tests for OSFT functionality."""

    def test_command_line_osft_params_validation(self):
        """Test that command line validates OSFT parameters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test script that validates OSFT params
            test_script = Path(tmpdir) / "validate_osft.py"
            test_script.write_text("""
import sys
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def main(
    osft: bool = False,
    osft_unfreeze_rank_ratio: Optional[float] = None,
    osft_target_patterns: Optional[str] = None
):
    # Validate: if osft is True, unfreeze_rank_ratio must be provided
    if osft and osft_unfreeze_rank_ratio is None:
        print("ERROR: osft_unfreeze_rank_ratio required")
        raise typer.Exit(1)

    # Parse target patterns if provided (comma-delimited)
    if osft_target_patterns:
        patterns = [p.strip() for p in osft_target_patterns.split(",")]
        print(f"PATTERNS: {patterns}")

    print(f"SUCCESS: osft={osft}, ratio={osft_unfreeze_rank_ratio}")

if __name__ == "__main__":
    app()
""")

            # Test valid OSFT configuration
            result = subprocess.run(
                [
                    "python",
                    str(test_script),
                    "--osft",
                    "--osft-unfreeze-rank-ratio=0.5",
                    "--osft-target-patterns=q_proj,k_proj",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "SUCCESS" in result.stdout
            assert "PATTERNS: ['q_proj', 'k_proj']" in result.stdout

            # Test missing unfreeze_rank_ratio
            result = subprocess.run(["python", str(test_script), "--osft"], capture_output=True, text=True)
            assert result.returncode == 1
            assert "ERROR: osft_unfreeze_rank_ratio required" in result.stdout

            # Test osft=False doesn't require rank_ratio
            result = subprocess.run(["python", str(test_script)], capture_output=True, text=True)
            assert result.returncode == 0
            assert "SUCCESS: osft=False" in result.stdout


class TestOSFTPrepareStateDict:
    """Test suite for OSFT prepare_state_dict_for_save functionality."""

    def test_osft_prepare_basic_reconstruction(self):
        """Test that OSFT parameters get reconstructed correctly."""

        # Create simple base model
        class SimpleModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.config = config or MagicMock()
                self.dtype = torch.float32

        # Create OSFT version
        OSFTModel = create_osft_model_class(SimpleModel)
        osft_config = {"linear.weight": 2}  # rank 2 decomposition

        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)

        # Get state dict with OSFT parameters
        osft_state_dict = model.state_dict()

        # Verify OSFT parameters exist
        osft_keys = [
            k
            for k in osft_state_dict.keys()
            if "osft_params" in k or "_U_high" in k or "_S_high" in k or "_V_high" in k
        ]
        assert len(osft_keys) > 0, "No OSFT parameters found"

        # Call prepare_state_dict_for_save
        reconstructed = model.prepare_state_dict_for_save(osft_state_dict.copy())

        # Verify OSFT parameters are removed and original weights are present
        assert "linear.weight" in reconstructed
        for key in reconstructed.keys():
            assert "osft_params" not in key
            assert "_U_high" not in key and "_S_high" not in key and "_V_high" not in key

        # Verify shape is correct
        assert reconstructed["linear.weight"].shape == (4, 4)

    def test_osft_prepare_preserves_non_osft(self):
        """Test that non-OSFT parameters are preserved unchanged."""

        class ModelWithNonOSFT(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.osft_layer = nn.Linear(4, 4, bias=False)
                self.regular_layer = nn.Linear(4, 2, bias=True)  # Will not be decomposed
                self.config = config or MagicMock()
                self.dtype = torch.float32

        OSFTModel = create_osft_model_class(ModelWithNonOSFT)
        osft_config = {"osft_layer.weight": 2}  # Only decompose one layer

        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)

        # Set known values for non-OSFT parameters
        with torch.no_grad():
            model.regular_layer.weight.fill_(3.14159)
            model.regular_layer.bias.fill_(2.71828)

        state_dict = model.state_dict()
        original_weight = state_dict["regular_layer.weight"].clone()
        original_bias = state_dict["regular_layer.bias"].clone()

        # Call prepare_state_dict_for_save
        reconstructed = model.prepare_state_dict_for_save(state_dict.copy())

        # Verify non-OSFT parameters are unchanged
        assert torch.equal(reconstructed["regular_layer.weight"], original_weight)
        assert torch.equal(reconstructed["regular_layer.bias"], original_bias)

    def test_osft_prepare_empty_config(self):
        """Test that models without OSFT work correctly."""

        class RegularModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.config = config or MagicMock()
                self.dtype = torch.float32

        OSFTModel = create_osft_model_class(RegularModel)
        model = OSFTModel(MagicMock(), osft_config={}, initialize_osft=False)

        state_dict = model.state_dict()
        original_keys = set(state_dict.keys())

        # Call prepare_state_dict_for_save (should be no-op)
        result = model.prepare_state_dict_for_save(state_dict.copy())

        # Verify state dict is unchanged
        assert set(result.keys()) == original_keys
        for key in original_keys:
            assert torch.equal(result[key], state_dict[key])

    def test_osft_prepare_dtype_preservation(self):
        """Test that reconstructed weights have correct dtype."""

        class TypedModel(nn.Module):
            def __init__(self, config=None, **kwargs):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.config = config or MagicMock()
                self.dtype = torch.float32
                self.output_dtype = torch.float32

        OSFTModel = create_osft_model_class(TypedModel)
        osft_config = {"linear.weight": 2}

        model = OSFTModel(MagicMock(), osft_config=osft_config, initialize_osft=False)
        model.reinitialize_osft(decompose_existing_weights=True)

        state_dict = model.state_dict()
        reconstructed = model.prepare_state_dict_for_save(state_dict.copy())

        # Verify dtype is preserved
        assert reconstructed["linear.weight"].dtype == torch.float32


class TestOSFTOrthogonality:
    """Test OSFT orthogonality constraints during training."""

    def _create_simple_osft_model(self, hidden_size=16, rank_ratio=0.5):
        """Create a simple model with OSFT for testing orthogonality."""

        class SimpleModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dtype = torch.float32

                # Initialize with reasonable values
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

        # Create OSFT version
        OSFTModelClass = create_osft_model_class(SimpleModel)

        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {"linear.weight": int(hidden_size * rank_ratio)}

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32,
        )

        # Store original weight before OSFT conversion
        original_weight = model.linear.weight.data.clone()

        # Set OSFT config and initialize
        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = rank_ratio
        model.reinitialize_osft(decompose_existing_weights=True)

        return model, original_weight

    def test_gradient_orthogonality_simple_model(self):
        """Test that gradients maintain orthogonality in a simple model."""

        model, _ = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        # Create input and target
        input_data = torch.randn(4, 16)
        target = torch.randn(4, 16)

        # Forward pass
        output = model.linear(input_data)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Project gradients to maintain orthogonality
        model.project_gradients()

        # Check gradient orthogonality
        for module in model.modules():
            if (
                hasattr(module, "osft_params")
                and hasattr(module, "osft_U_high")
                and hasattr(module, "osft_S_high")
                and hasattr(module, "osft_V_high")
            ):
                check_gradient_orthogonality(model, module, step=1, tracker=tracker)

        # Verify orthogonality is maintained
        assert tracker.is_successful(), f"Gradient orthogonality violated:\n{tracker.get_summary()}"

    def test_gradient_orthogonality_multi_layer(self):
        """Test gradient orthogonality with multiple OSFT layers."""

        class MultiLayerModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.layer1 = nn.Linear(16, 16, bias=False)
                self.layer2 = nn.Linear(16, 16, bias=False)
                self.layer3 = nn.Linear(16, 16, bias=False)
                self.dtype = torch.float32

                # Initialize weights
                for layer in [self.layer1, self.layer2, self.layer3]:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        OSFTModelClass = create_osft_model_class(MultiLayerModel)

        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {
            "layer1.weight": 8,
            "layer2.weight": 8,
            "layer3.weight": 8,
        }

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32,
        )

        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = 0.5
        model.reinitialize_osft(decompose_existing_weights=True)
        model.train()

        tracker = OrthogonalityTracker(margin_deg=1.0)

        # Forward and backward pass
        input_data = torch.randn(4, 16)
        x = model.layer1(input_data)
        x = model.layer2(x)
        x = model.layer3(x)
        target = torch.randn(4, 16)
        loss = torch.nn.functional.mse_loss(x, target)
        loss.backward()

        # Project gradients
        model.project_gradients()

        # Check gradient orthogonality for all layers
        for module in model.modules():
            if (
                hasattr(module, "osft_params")
                and hasattr(module, "osft_U_high")
                and hasattr(module, "osft_S_high")
                and hasattr(module, "osft_V_high")
            ):
                check_gradient_orthogonality(model, module, step=1, tracker=tracker)

        assert tracker.is_successful(), f"Multi-layer gradient orthogonality violated:\n{tracker.get_summary()}"

    def test_parameter_orthogonality_after_optimizer_step(self):
        """Test that parameters remain orthogonal after optimizer step."""

        model, _ = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        # Get OSFT parameters only
        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        assert len(osft_params) > 0
        optimizer = torch.optim.AdamW(osft_params, lr=1e-4)

        # Wrap optimizer to enable gradient projection
        optim_wrapper(optimizer, model)

        # Training step
        input_data = torch.randn(4, 16)
        target = torch.randn(4, 16)
        output = model.linear(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Project gradients
        model.project_gradients()

        # Check gradient orthogonality before optimizer step
        for module in model.modules():
            if (
                hasattr(module, "osft_params")
                and hasattr(module, "osft_U_high")
                and hasattr(module, "osft_S_high")
                and hasattr(module, "osft_V_high")
            ):
                check_gradient_orthogonality(model, module, step=1, tracker=tracker)

        # Optimizer step
        optimizer.step()

        # Check parameter orthogonality after optimizer step
        for module in model.modules():
            if (
                hasattr(module, "osft_params")
                and hasattr(module, "osft_U_high")
                and hasattr(module, "osft_S_high")
                and hasattr(module, "osft_V_high")
            ):
                check_parameter_orthogonality(model, module, step=1, tracker=tracker)

        assert tracker.is_successful(), (
            f"Parameter orthogonality violated after optimizer step:\n{tracker.get_summary()}"
        )

    def test_orthogonality_maintained_over_training(self):
        """Test that orthogonality is maintained over multiple training steps."""
        model, _ = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        assert len(osft_params) > 0
        optimizer = torch.optim.AdamW(osft_params, lr=1e-4)
        optim_wrapper(optimizer, model)

        num_steps = 20
        for step in range(1, num_steps + 1):
            # Generate random data
            input_data = torch.randn(4, 16)
            target = torch.randn(4, 16)

            # Forward pass
            output = model.linear(input_data)
            loss = torch.nn.functional.mse_loss(output, target)

            # Backward pass
            loss.backward()

            # Project gradients
            model.project_gradients()

            # Check gradient orthogonality
            for module in model.modules():
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    check_gradient_orthogonality(model, module, step, tracker)

            # Optimizer step
            optimizer.step()

            # Check parameter orthogonality
            for module in model.modules():
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    check_parameter_orthogonality(model, module, step, tracker)

            optimizer.zero_grad()

        assert tracker.is_successful(), f"Orthogonality violated during training:\n{tracker.get_summary()}"

    def test_orthogonality_with_different_rank_ratios(self):
        """Test orthogonality with different rank ratios."""

        rank_ratios = [0.1, 0.5, 0.9]

        for rank_ratio in rank_ratios:
            model, _ = self._create_simple_osft_model(hidden_size=16, rank_ratio=rank_ratio)
            tracker = OrthogonalityTracker(margin_deg=1.0)

            osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
            assert len(osft_params) > 0
            optimizer = torch.optim.AdamW(osft_params, lr=1e-4)
            optim_wrapper(optimizer, model)

            # Single training step
            input_data = torch.randn(4, 16)
            target = torch.randn(4, 16)
            output = model.linear(input_data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

            # Project gradients
            model.project_gradients()

            for module in model.modules():
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    check_gradient_orthogonality(model, module, step=1, tracker=tracker)

            optimizer.step()

            for module in model.modules():
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    check_parameter_orthogonality(model, module, step=1, tracker=tracker)

            assert tracker.is_successful(), f"Rank ratio {rank_ratio} failed orthogonality:\n{tracker.get_summary()}"

    def test_compute_angle_differences_utility(self):
        """Test the compute_angle_differences utility function."""
        # Test with perfectly orthogonal matrices
        # Create orthonormal columns using standard basis vectors
        torch.manual_seed(42)
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        B = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        B = B / B.norm(dim=0, keepdim=True)  # Normalize columns

        angles = compute_angle_differences(A, B, top_n=1)
        assert len(angles) > 0
        # A[:, 0] = [1, 0, 0] and B[:, 0] = [0, 0, 1] are orthogonal (90 deg)
        # So the deviation from orthogonality should be near 0
        assert angles[0] < 1.0, f"Expected small angle difference for orthogonal vectors, got {angles[0]}"

        # Test with non-orthogonal matrices
        C = torch.randn(10, 5)
        D = torch.randn(10, 3)

        angles = compute_angle_differences(C, D, top_n=3)
        assert len(angles) > 0
        # Random matrices likely won't be orthogonal, so we just check we get results
        assert len(angles) <= 3

        # Test with same matrix (self-orthogonality check)
        # Random matrices typically aren't self-orthogonal
        E = torch.randn(10, 5)
        angles = compute_angle_differences(E, None, top_n=3)
        assert len(angles) > 0

    def test_orthogonality_tracker(self):
        """Test the OrthogonalityTracker class."""
        tracker = OrthogonalityTracker(margin_deg=1.0)

        # Add some measurements
        tracker.update("param1", "U_grad", 0.5, step=1)
        tracker.update("param1", "V_grad", 0.3, step=1)
        tracker.update("param2", "U_grad", 1.5, step=2)  # Violation

        assert tracker.total_checks == 3
        assert tracker.failed_checks == 1
        assert not tracker.is_successful()

        # Check top violations
        violations = tracker.get_top_violations(n=3)
        assert len(violations) == 3
        assert violations[0]["max_angle_diff"] == 1.5

        # Test summary
        summary = tracker.get_summary()
        assert "FAILED" in summary
        assert "param2" in summary


class TestBatchedUAllReduce:
    """Test batched U projection all-reduce in project_gradients."""

    def _create_multi_target_model(self):
        """Create a model with multiple OSFT targets of different shapes."""

        class MultiLinearModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.small = nn.Linear(8, 8, bias=False)
                self.medium = nn.Linear(16, 8, bias=False)
                self.large = nn.Linear(16, 16, bias=False)
                self.dtype = torch.float32
                for layer in [self.small, self.medium, self.large]:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        OSFTModelClass = create_osft_model_class(MultiLinearModel)
        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {
            "small.weight": 4,
            "medium.weight": 4,
            "large.weight": 8,
        }

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float64,
            output_dtype=torch.float64,
        )
        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = 0.5
        model.reinitialize_osft(decompose_existing_weights=True)
        return model

    def test_skip_u_flag_leaves_u_gradient_unprojected(self):
        """skip_u=True should leave U_low gradient unchanged across all targets."""
        torch.manual_seed(42)
        model = self._create_multi_target_model()
        model.train()

        # Exercise all three targets so each has gradients
        x8 = torch.randn(2, 8, dtype=torch.float64)
        x16 = torch.randn(2, 16, dtype=torch.float64)
        loss = model.small(x8).pow(2).sum() + model.medium(x16).pow(2).sum() + model.large(x16).pow(2).sum()
        loss.backward()

        checked = 0
        for module in model.modules():
            if hasattr(module, "osft_V_high") and module.osft_params.U_low.grad is not None:
                grad_before = module.osft_params.U_low.grad.clone()
                svd_dict = model.get_svd_dict_for_module(module)
                project_gradient_to_orthogonal_space(svd_dict, skip_u=True)
                assert torch.equal(module.osft_params.U_low.grad, grad_before), (
                    "skip_u=True should not modify U_low gradient"
                )
                checked += 1
        assert checked == 3, f"Expected 3 targets with U gradients, got {checked}"

    def test_skip_u_still_projects_v(self):
        """skip_u=True should still project V_low gradient onto orthogonal complement."""
        torch.manual_seed(42)
        model = self._create_multi_target_model()
        model.train()

        # Exercise all targets
        x8 = torch.randn(2, 8, dtype=torch.float64)
        x16 = torch.randn(2, 16, dtype=torch.float64)
        loss = model.small(x8).pow(2).sum() + model.medium(x16).pow(2).sum() + model.large(x16).pow(2).sum()
        loss.backward()

        checked = 0
        for module in model.modules():
            if hasattr(module, "osft_V_high") and module.osft_params.V_low.grad is not None:
                svd_dict = model.get_svd_dict_for_module(module)
                project_gradient_to_orthogonal_space(svd_dict, skip_u=True)

                # Verify projected gradient is orthogonal to row(V_high):
                # dV @ V_high^T should be zero (each row of dV has no component
                # in the row space of V_high).
                dV = module.osft_params.V_low.grad
                V_high = module.osft_V_high
                overlap = torch.mm(dV, V_high.transpose(0, 1))
                assert torch.allclose(overlap, torch.zeros_like(overlap), atol=1e-10), (
                    f"Projected V gradient not orthogonal to V_high: max |dV @ V_high^T| = {overlap.abs().max():.2e}"
                )
                checked += 1
        assert checked == 3, f"Expected 3 targets with V gradients, got {checked}"

    def test_orthogonality_with_multiple_targets(self):
        """Orthogonality must hold for all targets after project_gradients."""
        model = self._create_multi_target_model()
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        optimizer = torch.optim.AdamW(osft_params, lr=1e-4)
        optim_wrapper(optimizer, model)

        for step in range(1, 6):
            x8 = torch.randn(2, 8, dtype=torch.float64)
            x16 = torch.randn(2, 16, dtype=torch.float64)
            # Exercise all three targets independently (different input dims)
            loss = model.small(x8).pow(2).sum() + model.medium(x16).pow(2).sum() + model.large(x16).pow(2).sum()
            loss.backward()
            optimizer.step()

            for module in model.modules():
                if hasattr(module, "osft_V_high") and hasattr(module, "osft_U_high") and hasattr(module, "osft_S_high"):
                    check_parameter_orthogonality(model, module, step, tracker)
            optimizer.zero_grad()

        assert tracker.is_successful(), f"Orthogonality violated with batched U all-reduce:\n{tracker.get_summary()}"

    def test_flatten_cat_split_roundtrip(self):
        """Verify that the flatten/cat/split logic preserves coefficient values."""
        shapes = [(4, 4), (4, 8), (8, 8)]
        originals = [torch.randn(s) for s in shapes]
        flat_parts = [c.flatten() for c in originals]
        batched = torch.cat(flat_parts)

        # Simulate split
        offset = 0
        for i, shape in enumerate(shapes):
            numel = shape[0] * shape[1]
            recovered = batched[offset : offset + numel].reshape(shape)
            offset += numel
            assert torch.equal(recovered, originals[i]), f"Round-trip failed for shape {shape}"

    def test_batched_path_matches_unbatched(self, monkeypatch):
        """The distributed batched path must produce identical gradients to the unbatched path.

        Mocks dist to force the batched code path with a no-op all-reduce
        (identity for single-rank), then compares against the non-distributed
        reference.  Uses identical seeds to create two models with the same
        initial state (avoids deepcopy issues with dynamic OSFT classes).
        """
        import torch.distributed as dist

        # Create two identical models from the same seed
        torch.manual_seed(99)
        model_ref = self._create_multi_target_model()
        model_ref.train()

        torch.manual_seed(99)
        model_bat = self._create_multi_target_model()
        model_bat.train()

        # Same input for both — generate AFTER model init so seed state is aligned
        x8 = torch.randn(2, 8, dtype=torch.float64)
        x16 = torch.randn(2, 16, dtype=torch.float64)

        # Reference: unbatched (non-distributed) path
        loss_ref = (
            model_ref.small(x8).pow(2).sum() + model_ref.medium(x16).pow(2).sum() + model_ref.large(x16).pow(2).sum()
        )
        loss_ref.backward()
        model_ref.project_gradients()  # takes non-distributed path

        # Batched: same forward/backward, then mock dist to force batched path
        loss_bat = (
            model_bat.small(x8).pow(2).sum() + model_bat.medium(x16).pow(2).sum() + model_bat.large(x16).pow(2).sum()
        )
        loss_bat.backward()

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        # No-op all_reduce: with a single real rank, the local value IS the
        # global value, so identity is correct.
        monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
        # V projection detects that local_V_high is already the full tensor
        # (shape[0] == rank_high) and skips all_gather, so no mock needed.
        model_bat.project_gradients()  # takes batched path

        # Compare every projected gradient
        for (n_ref, p_ref), (n_bat, p_bat) in zip(model_ref.named_parameters(), model_bat.named_parameters()):
            assert n_ref == n_bat
            if p_ref.grad is None and p_bat.grad is None:
                continue
            assert (p_ref.grad is None) == (p_bat.grad is None), (
                f"Gradient presence mismatch on {n_ref}: "
                f"ref={'present' if p_ref.grad is not None else 'None'}, "
                f"bat={'present' if p_bat.grad is not None else 'None'}"
            )
            assert torch.equal(p_ref.grad, p_bat.grad), (
                f"Gradient mismatch on {n_ref}: max |diff| = {(p_ref.grad - p_bat.grad).abs().max():.2e}"
            )


class TestVProjectionCache:
    """Test V_high caching in factored V projection."""

    def _create_simple_osft_model(self, hidden_size=32, rank_ratio=0.5):
        """Create a simple model with OSFT for testing."""

        class SimpleModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dtype = torch.float32
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

        OSFTModelClass = create_osft_model_class(SimpleModel)
        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {"linear.weight": int(hidden_size * rank_ratio)}

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = rank_ratio
        model.reinitialize_osft(decompose_existing_weights=True)
        return model

    def test_cache_populated_after_first_projection(self, monkeypatch):
        """V_high cache should be set on the module after the first call."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model()
        model.train()

        # Before projection: no cache
        for module in model.modules():
            if hasattr(module, "osft_params"):
                assert not hasattr(module, "_osft_v_high_full")

        # Forward + backward
        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()

        model.project_gradients()

        # After projection: cache should exist
        cached_count = 0
        for module in model.modules():
            if hasattr(module, "_osft_v_high_full"):
                cached_count += 1
                V = module._osft_v_high_full
                assert V.ndim == 2
                # Cached V_high should match V_high shape (k_high, M)
                assert V.shape == module.osft_V_high.shape
        assert cached_count > 0

    def test_cached_v_high_matches_original(self, monkeypatch):
        """Cached V_high should be identical to the original V_high (single GPU)."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model()
        model.train()

        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        for module in model.modules():
            if hasattr(module, "_osft_v_high_full") and hasattr(module, "osft_V_high"):
                cached = module._osft_v_high_full
                original = module.osft_V_high
                assert torch.equal(cached, original), "Cached V_high differs from original"

    def test_cache_smaller_than_gram(self, monkeypatch):
        """Cached V_high (k_high, M) should be smaller than the Gram matrix (M, M)."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model(hidden_size=32, rank_ratio=0.5)
        model.train()

        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        for module in model.modules():
            if hasattr(module, "_osft_v_high_full") and hasattr(module, "osft_V_high"):
                V = module._osft_v_high_full
                M = V.shape[1]
                cache_elements = V.nelement()  # k_high * M
                gram_elements = M * M  # M * M
                assert cache_elements < gram_elements, (
                    f"Cache ({cache_elements}) should be smaller than Gram ({gram_elements})"
                )

    def test_factored_matches_gram_projection(self):
        """Factored V projection should produce the same result as Gram-based projection."""
        torch.manual_seed(42)
        model = self._create_simple_osft_model(hidden_size=32, rank_ratio=0.5)
        model.train()

        # Get V_high for manual Gram-based projection
        V_high = None
        for module in model.modules():
            if hasattr(module, "osft_V_high"):
                V_high = module.osft_V_high.clone()
                break
        assert V_high is not None

        # Forward + backward
        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()

        # Save raw gradient before projection
        for module in model.modules():
            if hasattr(module, "osft_V_high") and hasattr(module, "osft_params"):
                raw_dV = module.osft_params.V_low.grad.clone()
                break

        # Apply factored projection (the actual code path)
        model.project_gradients()

        for module in model.modules():
            if hasattr(module, "osft_V_high") and hasattr(module, "osft_params"):
                factored_result = module.osft_params.V_low.grad.clone()
                break

        # Compute Gram-based projection manually for comparison
        G = V_high.T @ V_high
        gram_result = raw_dV - raw_dV @ G

        assert torch.allclose(factored_result, gram_result, atol=1e-6), (
            f"Factored and Gram projections differ: max diff = {(factored_result - gram_result).abs().max().item()}"
        )

    def test_factored_matches_gram_rectangular(self):
        """Factored and Gram agree for rectangular weights (the down_proj-like 7x case).

        For down_proj, N < M so k_high = min(N,M)/2 is small relative to M.
        This exercises the high-ratio regime where the factored form saves 7x.
        """
        torch.manual_seed(42)
        # Mimic down_proj shape ratio: N=16, M=56 → k_high=8, M=56 → ratio=7x
        N, M = 16, 56
        k_high = N // 2  # 8
        k_low = N - k_high  # 8

        # Create orthonormal V_high (k_high, M) via QR
        V_high = torch.linalg.qr(torch.randn(M, k_high))[0].T  # (k_high, M)
        assert V_high.shape == (k_high, M)

        dV = torch.randn(k_low, M)

        # Gram form: dV - dV @ (V_high^T @ V_high)
        G = V_high.T @ V_high  # (M, M) = (56, 56)
        gram_result = dV - dV @ G

        # Factored form: dV - (dV @ V_high^T) @ V_high
        coeff = dV @ V_high.T  # (k_low, k_high) = (8, 8)
        factored_result = dV - coeff @ V_high

        # Verify sizes confirm the 7x ratio
        assert G.nelement() == M * M  # 3136
        assert V_high.nelement() == k_high * M  # 448
        assert G.nelement() / V_high.nelement() == M / k_high  # 7.0

        assert torch.allclose(factored_result, gram_result, atol=1e-6), (
            f"Factored and Gram differ for rectangular case: max diff = "
            f"{(factored_result - gram_result).abs().max().item()}"
        )

    def test_projection_identical_with_and_without_cache(self, monkeypatch):
        """Projection results should be identical whether or not the cache is used."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        torch.manual_seed(42)
        model = self._create_simple_osft_model()
        model.train()

        # Step 1: populates cache
        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        optimizer = torch.optim.SGD(osft_params, lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()

        # Step 2: uses cache — save projected gradient
        x2 = torch.randn(4, 32)
        loss2 = model.linear(x2).pow(2).sum()
        loss2.backward()
        model.project_gradients()

        grad_with_cache = {}
        for module in model.modules():
            if hasattr(module, "osft_V_high"):
                grad_with_cache["V_low"] = module.osft_params.V_low.grad.clone()
                grad_with_cache["U_low"] = module.osft_params.U_low.grad.clone()

        # Now clear cache and redo the same step
        optimizer.zero_grad()
        for module in model.modules():
            if hasattr(module, "_osft_v_high_full"):
                del module._osft_v_high_full

        loss2b = model.linear(x2).pow(2).sum()
        loss2b.backward()
        model.project_gradients()

        for module in model.modules():
            if hasattr(module, "osft_V_high"):
                assert torch.equal(module.osft_params.V_low.grad, grad_with_cache["V_low"]), (
                    "V_low gradient differs with vs without cache"
                )
                assert torch.equal(module.osft_params.U_low.grad, grad_with_cache["U_low"]), (
                    "U_low gradient differs with vs without cache"
                )

    def test_cache_disabled_by_default(self, monkeypatch):
        """Cache should not be populated when OSFT_CACHE_V is at its default (off)."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", False)

        model = self._create_simple_osft_model()
        model.train()

        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        for module in model.modules():
            assert not hasattr(module, "_osft_v_high_full"), (
                "Cache should not be populated when OSFT_CACHE_V is disabled"
            )

    def test_orthogonality_maintained_with_cache(self, monkeypatch):
        """Orthogonality must hold across multiple steps with caching active."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model()
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        optimizer = torch.optim.AdamW(osft_params, lr=1e-4)
        optim_wrapper(optimizer, model)

        for step in range(1, 11):
            x = torch.randn(4, 32)
            loss = model.linear(x).pow(2).sum()
            loss.backward()

            # project_gradients is called inside optim_wrapper's step()
            optimizer.step()

            for module in model.modules():
                if (
                    hasattr(module, "osft_params")
                    and hasattr(module, "osft_U_high")
                    and hasattr(module, "osft_S_high")
                    and hasattr(module, "osft_V_high")
                ):
                    check_parameter_orthogonality(model, module, step, tracker)

            optimizer.zero_grad()

        assert tracker.is_successful(), f"Orthogonality violated with V_high caching:\n{tracker.get_summary()}"

    def test_cache_cleared_by_reset_osft_metadata(self, monkeypatch):
        """_reset_osft_metadata must clear cached V_high tensors.

        reinitialize_osft calls _reset_osft_metadata, which creates new V_high
        tensors.  Any cached all-gathered V_high from the old decomposition
        would be stale.  This test verifies the cache is cleared.
        """
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model()
        model.train()

        # Populate cache
        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        # Verify cache exists
        cached_modules = [m for m in model.modules() if hasattr(m, "_osft_v_high_full")]
        assert len(cached_modules) > 0, "Cache should be populated before reset"

        # _reset_osft_metadata is the mechanism reinitialize_osft uses
        model._reset_osft_metadata()

        # Cache must be gone
        for module in model.modules():
            assert not hasattr(module, "_osft_v_high_full"), "Cache was not cleared by _reset_osft_metadata"

    def test_cache_not_in_state_dict(self, monkeypatch):
        """Cached V_high tensors must not appear in model state_dict."""
        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)
        model = self._create_simple_osft_model()
        model.train()

        x = torch.randn(4, 32)
        loss = model.linear(x).pow(2).sum()
        loss.backward()
        model.project_gradients()

        # Verify cache exists
        assert any(hasattr(m, "_osft_v_high_full") for m in model.modules())

        # Verify it's not in state_dict
        sd = model.state_dict()
        for key in sd:
            assert "_osft_v_high_full" not in key, f"Cache leaked into state_dict: {key}"


class TestUnevenShardDeinterleave:
    """Test the uneven-shard de-interleave path in factored V projection.

    When k_high % world_size != 0, DTensor Shard(0) uses torch.chunk
    semantics: all shards have ceil(k_high/ws) rows except the last
    which gets the remainder.  all_gather_into_tensor requires equal-
    sized inputs, so each shard is padded to ceil rows.  After gathering,
    padding zeros are interspersed (not at the tail) and must be
    extracted per rank.

    These tests mock the distributed primitives to exercise the de-interleave
    logic from a single-process test.
    """

    @staticmethod
    def _make_ortho_V(k_high, M):
        """Create an orthonormal (k_high, M) matrix via QR."""
        Q = torch.linalg.qr(torch.randn(M, k_high, dtype=torch.float64))[0]
        return Q.T[:k_high].contiguous()  # (k_high, M)

    @staticmethod
    def _shard_dtensor_style(V_full, world_size):
        """Split V_full (k_high, M) into per-rank shards using torch.chunk semantics.

        DTensor Shard(0) uses torch.chunk: all shards have ceil(k/ws) rows
        except the last which gets the remainder.
        """
        return list(torch.chunk(V_full, world_size, dim=0))

    def _run_uneven_shard_test(self, k_high, world_size, monkeypatch):
        """Run one uneven-shard test case.

        Simulates rank 0's view: its local shard is chunk[0].
        The mock all_gather fills the output buffer with all ranks' padded shards.
        Then project_gradient_to_orthogonal_space de-interleaves and projects.
        """
        import math

        import torch.distributed as dist

        M = k_high * 3  # arbitrary, just needs M > k_high
        k_low = max(k_high // 2, 1)

        torch.manual_seed(42)
        V_full = self._make_ortho_V(k_high, M)
        dV_raw = torch.randn(k_low, M, dtype=torch.float64)

        # Reference: non-distributed factored projection
        coeff_ref = dV_raw @ V_full.T
        dV_ref = dV_raw - coeff_ref @ V_full

        # Shards as DTensor Shard(0) would produce
        shards = self._shard_dtensor_style(V_full, world_size)
        local_shard = shards[0]  # rank 0's shard

        # Build mock all_gather: simulate all ranks contributing padded shards
        rows_per_rank = math.ceil(k_high / world_size)

        def mock_all_gather(output, input_tensor):
            """Fill output with padded shards from all ranks."""
            for i, shard in enumerate(shards):
                padded = torch.zeros(rows_per_rank, M, dtype=V_full.dtype, device=V_full.device)
                padded[: shard.shape[0]].copy_(shard)
                output[i * rows_per_rank : (i + 1) * rows_per_rank].copy_(padded)

        # Create a V_high tensor with to_local() returning the local shard
        class FakeShardedTensor:
            """Mimics a DTensor with to_local() returning the local shard."""

            def __init__(self, local):
                self._local = local

            def to_local(self):
                return self._local

            # Proxy attribute access to the local tensor for anything else
            def __getattr__(self, name):
                return getattr(self._local, name)

        fake_V_high = FakeShardedTensor(local_shard)

        # Build svd_dict — only V_low grad and V_high matter for skip_u=True
        V_low = nn.Parameter(torch.randn(k_low, M, dtype=torch.float64))
        V_low.grad = dV_raw.clone()

        svd_dict = {
            "U_high": torch.randn(1, 1, dtype=torch.float64),  # unused with skip_u
            "S_high": torch.randn(k_high, dtype=torch.float64),
            "V_high": fake_V_high,
            "U_low": nn.Parameter(torch.randn(1, 1, dtype=torch.float64)),
            "S_low": nn.Parameter(torch.randn(1, dtype=torch.float64)),
            "V_low": V_low,
            "rank_high": k_high,
        }

        # Mock distributed
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
        monkeypatch.setattr(dist, "all_gather_into_tensor", mock_all_gather)

        project_gradient_to_orthogonal_space(svd_dict, skip_u=True)

        projected_dV = V_low.grad
        assert torch.allclose(projected_dV, dV_ref, atol=1e-10), (
            f"Uneven shard de-interleave failed for k_high={k_high}, ws={world_size}: "
            f"max |diff| = {(projected_dV - dV_ref).abs().max().item():.2e}"
        )

        # Verify orthogonality: projected gradient has no component in row(V_high)
        overlap = projected_dV @ V_full.T
        assert torch.allclose(overlap, torch.zeros_like(overlap), atol=1e-10), (
            f"Projected gradient not orthogonal to V_high: max |dV @ V_high^T| = {overlap.abs().max().item():.2e}"
        )

    @pytest.mark.parametrize(
        "k_high,world_size",
        [
            (7, 3),  # chunk sizes [3, 3, 1]
            (5, 3),  # chunk sizes [2, 2, 1]
            (10, 3),  # chunk sizes [4, 4, 2]
            (11, 4),  # chunk sizes [3, 3, 3, 2]
            (3, 2),  # chunk sizes [2, 1]
        ],
    )
    def test_uneven_shard_cases(self, k_high, world_size, monkeypatch):
        """Uneven shard de-interleave must recover the correct V_high."""
        self._run_uneven_shard_test(k_high, world_size, monkeypatch)

    @pytest.mark.parametrize(
        "k_high,world_size",
        [
            (6, 3),  # even: shards [2, 2, 2]
            (8, 4),  # even: shards [2, 2, 2, 2]
            (10, 2),  # even: shards [5, 5]
        ],
    )
    def test_even_shard_baseline(self, k_high, world_size, monkeypatch):
        """Even shards (no remainder) should also work correctly."""
        self._run_uneven_shard_test(k_high, world_size, monkeypatch)

    def test_cache_stores_deinterleaved_v_high(self, monkeypatch):
        """Cache must store the correctly de-interleaved V_high, not raw gathered buffer.

        Step 1 populates the cache via the uneven all-gather path.
        Step 2 hits the cache (no all-gather).  Both must produce the
        same projection as the non-distributed reference.
        """
        import math

        import torch.distributed as dist

        monkeypatch.setattr(osft_module, "OSFT_CACHE_V", True)

        k_high, world_size = 7, 3  # uneven: chunk sizes [3, 3, 1]
        M = k_high * 3
        k_low = k_high // 2

        torch.manual_seed(42)
        V_full = self._make_ortho_V(k_high, M)
        shards = self._shard_dtensor_style(V_full, world_size)
        local_shard = shards[0]
        rows_per_rank = math.ceil(k_high / world_size)

        gather_call_count = 0

        def mock_all_gather(output, input_tensor):
            nonlocal gather_call_count
            gather_call_count += 1
            for i, shard in enumerate(shards):
                padded = torch.zeros(rows_per_rank, M, dtype=V_full.dtype, device=V_full.device)
                padded[: shard.shape[0]].copy_(shard)
                output[i * rows_per_rank : (i + 1) * rows_per_rank].copy_(padded)

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
        monkeypatch.setattr(dist, "all_gather_into_tensor", mock_all_gather)

        class FakeShardedTensor:
            def __init__(self, local):
                self._local = local

            def to_local(self):
                return self._local

            def __getattr__(self, name):
                return getattr(self._local, name)

        cache_holder = nn.Module()

        results = []
        for step in range(2):
            dV_raw = torch.randn(k_low, M, dtype=torch.float64)
            dV_ref = dV_raw - (dV_raw @ V_full.T) @ V_full

            V_low = nn.Parameter(torch.randn(k_low, M, dtype=torch.float64))
            V_low.grad = dV_raw.clone()

            svd_dict = {
                "U_high": torch.randn(1, 1, dtype=torch.float64),
                "S_high": torch.randn(k_high, dtype=torch.float64),
                "V_high": FakeShardedTensor(local_shard),
                "U_low": nn.Parameter(torch.randn(1, 1, dtype=torch.float64)),
                "S_low": nn.Parameter(torch.randn(1, dtype=torch.float64)),
                "V_low": V_low,
                "rank_high": k_high,
            }

            project_gradient_to_orthogonal_space(svd_dict, skip_u=True, cache_holder=cache_holder)
            results.append((V_low.grad.clone(), dV_ref))

        # Step 1 should have called all_gather; step 2 should have used cache
        assert gather_call_count == 1, f"Expected 1 all_gather call, got {gather_call_count}"
        assert hasattr(cache_holder, "_osft_v_high_full"), "Cache was not populated"

        # Both steps must match reference
        for step, (projected, ref) in enumerate(results):
            assert torch.allclose(projected, ref, atol=1e-10), (
                f"Step {step}: projection wrong, max |diff| = {(projected - ref).abs().max().item():.2e}"
            )

        # Cached tensor must equal the full V_high (not the padded gathered buffer)
        assert cache_holder._osft_v_high_full.shape == (k_high, M), (
            f"Cached shape {cache_holder._osft_v_high_full.shape} != expected ({k_high}, {M})"
        )
        assert torch.allclose(cache_holder._osft_v_high_full, V_full, atol=1e-10), (
            "Cached V_high differs from original — may contain padding"
        )


class TestLazyInitTokenizerAlignment:
    """Ensure memory-efficient loading aligns tokenizers before broadcasting."""

    def test_memory_efficient_loading_calls_alignment_hook(self, monkeypatch):
        """Alignment hook should run on the fully materialized CPU model."""
        loaded_models = []

        class DummyLoadedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.vocab_size = 10
                self.aligned = False

            def state_dict(self):
                return {"weight": torch.zeros(1)}

            def named_buffers(self):
                return [("buffer", torch.zeros(1))]

        class DummyBase(nn.Module):
            def __init__(self):
                super().__init__()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                model = DummyLoadedModel()
                loaded_models.append(model)
                return model

        class DummyOSFT(DummyBase):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self._lazy_init_pending = True

        def _align(model):
            model.aligned = True
            return model

        align_mock = MagicMock(side_effect=_align)

        monkeypatch.setattr(osft_module.dist, "is_available", lambda: True)
        monkeypatch.setattr(osft_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(osft_module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(osft_module.dist, "barrier", lambda: None)
        monkeypatch.setattr(osft_module.dist, "broadcast_object_list", lambda *_, **__: None)
        monkeypatch.setattr(osft_module.torch.cuda, "is_available", lambda: False)

        model = _load_model_memory_efficient(
            actual_osft_cls=DummyOSFT,
            pretrained_model_name_or_path="dummy",
            model_args=tuple(),
            base_kwargs={"torch_dtype": torch.float32},
            osft_class_kwargs={"lazy_init_tokenizer_align_fn": align_mock},
        )

        assert isinstance(model, DummyOSFT)
        assert align_mock.call_count == 1
        assert loaded_models and loaded_models[0].aligned is True

    def test_memory_efficient_loading_train_dtype_overrides_torch_dtype(self, monkeypatch):
        """train_dtype should take precedence over torch_dtype in base_kwargs."""
        captured_kwargs = {}

        class DummyLoadedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.vocab_size = 10

            def state_dict(self):
                return {"weight": torch.zeros(1)}

            def named_buffers(self):
                return [("buffer", torch.zeros(1))]

        class DummyBase(nn.Module):
            def __init__(self):
                super().__init__()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured_kwargs.update(kwargs)
                return DummyLoadedModel()

        class DummyOSFT(DummyBase):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self._lazy_init_pending = True

        monkeypatch.setattr(osft_module.dist, "is_available", lambda: True)
        monkeypatch.setattr(osft_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(osft_module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(osft_module.dist, "barrier", lambda: None)
        monkeypatch.setattr(osft_module.dist, "broadcast_object_list", lambda *_, **__: None)
        monkeypatch.setattr(osft_module.torch.cuda, "is_available", lambda: False)

        model = _load_model_memory_efficient(
            actual_osft_cls=DummyOSFT,
            pretrained_model_name_or_path="dummy",
            model_args=tuple(),
            base_kwargs={"torch_dtype": torch.float32},
            osft_class_kwargs={},
            train_dtype=torch.bfloat16,
        )

        assert isinstance(model, DummyOSFT)
        assert captured_kwargs["torch_dtype"] == torch.bfloat16

    def test_memory_efficient_loading_falls_back_to_torch_dtype(self, monkeypatch):
        """When train_dtype is None, should fall back to torch_dtype from base_kwargs."""
        captured_kwargs = {}

        class DummyLoadedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.vocab_size = 10

            def state_dict(self):
                return {"weight": torch.zeros(1)}

            def named_buffers(self):
                return [("buffer", torch.zeros(1))]

        class DummyBase(nn.Module):
            def __init__(self):
                super().__init__()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured_kwargs.update(kwargs)
                return DummyLoadedModel()

        class DummyOSFT(DummyBase):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self._lazy_init_pending = True

        monkeypatch.setattr(osft_module.dist, "is_available", lambda: True)
        monkeypatch.setattr(osft_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(osft_module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(osft_module.dist, "barrier", lambda: None)
        monkeypatch.setattr(osft_module.dist, "broadcast_object_list", lambda *_, **__: None)
        monkeypatch.setattr(osft_module.torch.cuda, "is_available", lambda: False)

        model = _load_model_memory_efficient(
            actual_osft_cls=DummyOSFT,
            pretrained_model_name_or_path="dummy",
            model_args=tuple(),
            base_kwargs={"torch_dtype": torch.float16},
            osft_class_kwargs={},
        )

        assert isinstance(model, DummyOSFT)
        assert captured_kwargs["torch_dtype"] == torch.float16


class TestPostStepParameterProjection:
    """Test post-step parameter re-projection to fix AdamW subspace leak.

    AdamW's element-wise moment rescaling (m̂_t / √v̂_t) can rotate the
    parameter update out of the orthogonal complement of the frozen
    subspace, even when gradients are correctly projected beforehand.
    The ``project_parameters`` method and updated ``optim_wrapper``
    re-project parameters after each step to correct this.
    """

    def _create_simple_osft_model(self, hidden_size=16, rank_ratio=0.5):
        """Create a simple model with OSFT for testing."""

        class SimpleModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dtype = torch.float32
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

        OSFTModelClass = create_osft_model_class(SimpleModel)
        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {"linear.weight": int(hidden_size * rank_ratio)}

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = rank_ratio
        model.reinitialize_osft(decompose_existing_weights=True)
        return model

    def _create_multi_target_model(self):
        """Create a model with multiple OSFT targets of different shapes."""

        class MultiLinearModel(nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.small = nn.Linear(8, 8, bias=False)
                self.medium = nn.Linear(16, 8, bias=False)
                self.large = nn.Linear(16, 16, bias=False)
                self.dtype = torch.float32
                for layer in [self.small, self.medium, self.large]:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        OSFTModelClass = create_osft_model_class(MultiLinearModel)
        config = MagicMock()
        config.vocab_size = 1000
        osft_config = {
            "small.weight": 4,
            "medium.weight": 4,
            "large.weight": 8,
        }

        model = OSFTModelClass(
            config=config,
            osft_config={},
            initialize_osft=False,
            upcast_dtype=torch.float64,
            output_dtype=torch.float64,
        )
        model.osft_config = osft_config
        model.osft_unfreeze_rank_ratio = 0.5
        model.reinitialize_osft(decompose_existing_weights=True)
        return model

    def test_project_parameters_basic(self):
        """Test that project_parameters projects U_low and V_low correctly."""
        model = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)

        # Manually perturb U_low and V_low to have components in frozen subspace
        for module in model.modules():
            if hasattr(module, "osft_params") and hasattr(module, "osft_U_high"):
                U_high = module.osft_U_high
                V_high = module.osft_V_high

                # Add a component in the frozen subspace direction
                with torch.no_grad():
                    module.osft_params.U_low.data += U_high @ torch.randn(
                        U_high.shape[1], module.osft_params.U_low.shape[1]
                    )
                    module.osft_params.V_low.data += (
                        torch.randn(module.osft_params.V_low.shape[0], V_high.shape[0]) @ V_high
                    )

        # Now project parameters
        model.project_parameters()

        # Check that U_low and V_low are orthogonal to frozen subspace
        for module in model.modules():
            if hasattr(module, "osft_params") and hasattr(module, "osft_U_high"):
                U_high = module.osft_U_high
                V_high = module.osft_V_high
                U_low = module.osft_params.U_low
                V_low = module.osft_params.V_low

                # U_high^T @ U_low should be zero
                u_overlap = torch.mm(U_high.transpose(0, 1), U_low.data)
                assert torch.allclose(u_overlap, torch.zeros_like(u_overlap), atol=1e-5), (
                    f"U_low not orthogonal to U_high after projection: max |U_high^T @ U_low| = {u_overlap.abs().max():.2e}"
                )

                # V_low @ V_high^T should be zero
                v_overlap = torch.mm(V_low.data, V_high.transpose(0, 1))
                assert torch.allclose(v_overlap, torch.zeros_like(v_overlap), atol=1e-5), (
                    f"V_low not orthogonal to V_high after projection: max |V_low @ V_high^T| = {v_overlap.abs().max():.2e}"
                )

    def test_project_parameter_to_orthogonal_space_directly(self):
        """Test project_parameter_to_orthogonal_space helper directly."""
        torch.manual_seed(42)
        N, M = 16, 16
        k_high = 8
        k_low = 8

        # Create orthonormal U_high and V_high via QR
        U_high = torch.linalg.qr(torch.randn(N, k_high))[0]
        V_high = torch.linalg.qr(torch.randn(M, k_high))[0].T

        # Create U_low and V_low with components in frozen subspace
        U_low_data = torch.randn(N, k_low) + U_high @ torch.randn(k_high, k_low)
        V_low_data = torch.randn(k_low, M) + torch.randn(k_low, k_high) @ V_high

        U_low = nn.Parameter(U_low_data)
        V_low = nn.Parameter(V_low_data)

        svd_dict = {
            "U_high": U_high,
            "S_high": torch.ones(k_high),
            "V_high": V_high,
            "U_low": U_low,
            "S_low": nn.Parameter(torch.ones(k_low)),
            "V_low": V_low,
            "rank_high": k_high,
        }

        project_parameter_to_orthogonal_space(svd_dict)

        # Verify orthogonality (atol=1e-5 accounts for float32 accumulation)
        u_overlap = torch.mm(U_high.T, U_low.data)
        assert torch.allclose(u_overlap, torch.zeros_like(u_overlap), atol=1e-5), (
            f"U_low not orthogonal: max |U_high^T @ U_low| = {u_overlap.abs().max():.2e}"
        )

        v_overlap = torch.mm(V_low.data, V_high.T)
        assert torch.allclose(v_overlap, torch.zeros_like(v_overlap), atol=1e-5), (
            f"V_low not orthogonal: max |V_low @ V_high^T| = {v_overlap.abs().max():.2e}"
        )

    def test_adamw_leak_fixed_over_many_steps(self):
        """Stress test: verify orthogonality over 200 steps with aggressive LR.

        This test would fail without post-step parameter re-projection
        because AdamW's element-wise rescaling accumulates drift into the
        frozen subspace over many steps.
        """
        torch.manual_seed(42)
        model = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        assert len(osft_params) > 0
        optimizer = torch.optim.AdamW(osft_params, lr=1e-2)
        optim_wrapper(optimizer, model)

        num_steps = 200
        for step in range(1, num_steps + 1):
            input_data = torch.randn(4, 16)
            target = torch.randn(4, 16)

            output = model.linear(input_data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Check parameter orthogonality every 10 steps
            if step % 10 == 0:
                for module in model.modules():
                    if (
                        hasattr(module, "osft_params")
                        and hasattr(module, "osft_U_high")
                        and hasattr(module, "osft_S_high")
                        and hasattr(module, "osft_V_high")
                    ):
                        check_parameter_orthogonality(model, module, step, tracker)

        assert tracker.is_successful(), (
            f"AdamW subspace leak detected after {num_steps} steps:\n{tracker.get_summary()}"
        )

    def test_optim_wrapper_calls_project_parameters(self):
        """Test that the optim_wrapper calls project_parameters after step."""
        model = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)
        model.train()

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        optimizer = torch.optim.AdamW(osft_params, lr=1e-3)
        optim_wrapper(optimizer, model)

        # Track calls to project_parameters
        calls = []
        original_project_parameters = model.project_parameters

        def tracking_project_parameters():
            calls.append("project_parameters")
            return original_project_parameters()

        model.project_parameters = tracking_project_parameters

        # Do a step
        input_data = torch.randn(4, 16)
        target = torch.randn(4, 16)
        output = model.linear(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        assert "project_parameters" in calls, "optim_wrapper should call project_parameters after step"

    def test_project_parameters_multi_target(self):
        """Test post-step projection works with multiple OSFT targets."""
        model = self._create_multi_target_model()
        model.train()
        tracker = OrthogonalityTracker(margin_deg=1.0)

        osft_params = [p for n, p in model.named_parameters() if "osft_params" in n]
        optimizer = torch.optim.AdamW(osft_params, lr=1e-2)
        optim_wrapper(optimizer, model)

        for step in range(1, 51):
            x8 = torch.randn(2, 8, dtype=torch.float64)
            x16 = torch.randn(2, 16, dtype=torch.float64)
            loss = model.small(x8).pow(2).sum() + model.medium(x16).pow(2).sum() + model.large(x16).pow(2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 5 == 0:
                for module in model.modules():
                    if (
                        hasattr(module, "osft_V_high")
                        and hasattr(module, "osft_U_high")
                        and hasattr(module, "osft_S_high")
                    ):
                        check_parameter_orthogonality(model, module, step, tracker)

        assert tracker.is_successful(), f"Multi-target orthogonality violated:\n{tracker.get_summary()}"

    def test_project_parameters_idempotent(self):
        """Test that project_parameters is idempotent (applying twice yields same result)."""
        model = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)

        # Perturb parameters
        for module in model.modules():
            if hasattr(module, "osft_params") and hasattr(module, "osft_U_high"):
                with torch.no_grad():
                    module.osft_params.U_low.data += module.osft_U_high @ torch.randn(
                        module.osft_U_high.shape[1], module.osft_params.U_low.shape[1]
                    )

        # First projection
        model.project_parameters()
        first_state = {}
        for n, p in model.named_parameters():
            if "osft_params" in n:
                first_state[n] = p.data.clone()

        # Second projection (should be near-no-op; float32 accumulation
        # introduces ~1e-6 residual, so we use atol=1e-5)
        model.project_parameters()
        for n, p in model.named_parameters():
            if "osft_params" in n:
                assert torch.allclose(p.data, first_state[n], atol=1e-5), (
                    f"project_parameters is not idempotent for {n}: "
                    f"max diff = {(p.data - first_state[n]).abs().max():.2e}"
                )

    def test_project_parameters_preserves_s_low(self):
        """Test that project_parameters does not modify S_low."""
        model = self._create_simple_osft_model(hidden_size=16, rank_ratio=0.5)

        # Save S_low values
        s_low_before = {}
        for n, p in model.named_parameters():
            if "S_low" in n:
                s_low_before[n] = p.data.clone()

        model.project_parameters()

        for n, p in model.named_parameters():
            if "S_low" in n:
                assert torch.equal(p.data, s_low_before[n]), f"project_parameters modified S_low: {n}"

    def test_batched_param_projection_matches_unbatched(self, monkeypatch):
        """Distributed batched path must produce identical results to unbatched.

        Mirrors TestBatchedUAllReduce.test_batched_path_matches_unbatched
        but for parameter projection instead of gradient projection.
        """
        import torch.distributed as dist

        torch.manual_seed(99)
        model_ref = self._create_multi_target_model()

        torch.manual_seed(99)
        model_bat = self._create_multi_target_model()

        # Perturb both identically
        torch.manual_seed(42)
        for module in model_ref.modules():
            if hasattr(module, "osft_params") and hasattr(module, "osft_U_high"):
                perturbation_u = torch.randn_like(module.osft_params.U_low.data) * 0.1
                perturbation_v = torch.randn_like(module.osft_params.V_low.data) * 0.1
                with torch.no_grad():
                    module.osft_params.U_low.data += perturbation_u
                    module.osft_params.V_low.data += perturbation_v

        torch.manual_seed(42)
        for module in model_bat.modules():
            if hasattr(module, "osft_params") and hasattr(module, "osft_U_high"):
                perturbation_u = torch.randn_like(module.osft_params.U_low.data) * 0.1
                perturbation_v = torch.randn_like(module.osft_params.V_low.data) * 0.1
                with torch.no_grad():
                    module.osft_params.U_low.data += perturbation_u
                    module.osft_params.V_low.data += perturbation_v

        # Reference: unbatched (non-distributed) path
        model_ref.project_parameters()

        # Batched: mock dist to force batched path
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
        model_bat.project_parameters()

        # Compare every parameter
        for (n_ref, p_ref), (n_bat, p_bat) in zip(model_ref.named_parameters(), model_bat.named_parameters()):
            assert n_ref == n_bat
            assert torch.equal(p_ref.data, p_bat.data), (
                f"Parameter mismatch on {n_ref}: max |diff| = {(p_ref.data - p_bat.data).abs().max():.2e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
