"""Test mixed precision training to ensure proper dtype handling."""

import os

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from mini_trainer import device_utils
from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import patch_target_module


def create_tiny_llama_model():
    """Create a tiny Llama model with <1M parameters."""
    config = LlamaConfig(
        vocab_size=1000,  # Small vocabulary to keep params <1M
        hidden_size=64,  # Tiny hidden size
        intermediate_size=128,  # Small FFN
        num_hidden_layers=2,  # Only 2 layers
        num_attention_heads=4,  # Few attention heads
        num_key_value_heads=2,  # GQA
        max_position_embeddings=128,  # Short sequences
        rope_theta=10000.0,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(config)

    # Count parameters to ensure it's <1M
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    assert total_params < 1_000_000, f"Model has {total_params:,} params, should be <1M"

    return model, config


def check_tensor_dtype(tensor, expected_dtype, name):
    """Helper to check tensor dtype and report mismatches."""
    if tensor is None:
        return True

    actual_dtype = tensor.dtype
    if actual_dtype != expected_dtype:
        print(f"❌ {name}: Expected {expected_dtype}, got {actual_dtype}")
        return False
    else:
        print(f"✅ {name}: Correct dtype {actual_dtype}")
        return True


@pytest.mark.gpu
class TestMixedPrecisionDtypes:
    """Test that mixed precision training maintains proper dtypes."""

    def test_fsdp2_mixed_precision_dtypes(self, tmp_path, single_gpu_device):
        """
        Test that with FSDP2 mixed precision:
        - Model parameters are in FP32 (not BF16)
        - Gradients are in FP32 (not BF16)
        - Optimizer states are in FP32 (not BF16)

        This test catches the bug where reduce_dtype was incorrectly set to BF16
        instead of FP32 in the MixedPrecisionPolicy.
        """
        # Initialize distributed environment for FSDP2
        if not dist.is_initialized():
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend=device_utils.get_dist_backend(), rank=0, world_size=1)

        try:
            # Create tiny model
            model, config = create_tiny_llama_model()

            # Create a minimal tokenizer (just for alignment)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            # Save model and tokenizer to temp directory (required by setup_model)
            model_path = tmp_path / "tiny_llama_model"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            # Patch loss function for none reduction
            from mini_trainer.none_reduction_losses import hf_fixed_cross_entropy_none_reduction

            patch_target_module(
                "transformers.loss.loss_utils.fixed_cross_entropy",
                hf_fixed_cross_entropy_none_reduction,
            )

            # Load model through setup_model and setup_training_components
            # This applies FSDP2 wrapping with the mixed precision policy
            model = setup_model(
                model_name_or_path=str(model_path),
                use_liger_kernels=False,
                osft=False,
                local_rank=0,
            )

            # Setup training components (includes FSDP2 wrapping)
            model, optimizer, lr_scheduler = setup_training_components(
                model, learning_rate=1e-3, num_warmup_steps=0, lr_scheduler="constant"
            )

            # Create mock input for a simple forward/backward pass
            batch_size = 2
            seq_length = 32
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(single_gpu_device)
            labels = input_ids.clone()

            # Run a simple forward/backward pass
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            # Handle potentially unreduced loss
            loss = loss.float().sum() / batch_size

            # Backward pass
            loss.backward()

            # Optimizer step to create optimizer states
            optimizer.step()

            print(f"\n{'=' * 60}")
            print("DTYPE VERIFICATION AFTER FSDP2 WRAPPING AND TRAINING STEP")
            print(f"{'=' * 60}\n")

            # Now check dtypes of model parameters, gradients, and optimizer states
            all_checks_passed = True

            # Check model parameters
            print("Checking Model Parameters:")
            param_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # With FSDP2, the underlying parameter storage should be FP32
                    # even if computations happen in BF16
                    expected_dtype = torch.float32
                    passed = check_tensor_dtype(param, expected_dtype, f"  Parameter '{name}'")
                    all_checks_passed = all_checks_passed and passed

                    param_count += 1
                    if param_count >= 3:  # Check first few parameters
                        break

            print("\nChecking Gradients:")
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Gradients should be in FP32 for proper accumulation
                    expected_dtype = torch.float32
                    passed = check_tensor_dtype(param.grad, expected_dtype, f"  Gradient '{name}'")
                    all_checks_passed = all_checks_passed and passed

                    grad_count += 1
                    if grad_count >= 3:  # Check first few gradients
                        break

            print("\nChecking Optimizer States:")
            # Check optimizer state dtypes
            state_checked = False
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in optimizer.state:
                        state = optimizer.state[p]

                        # AdamW maintains exp_avg and exp_avg_sq in FP32
                        if "exp_avg" in state:
                            passed = check_tensor_dtype(
                                state["exp_avg"],
                                torch.float32,
                                "  Optimizer state 'exp_avg'",
                            )
                            all_checks_passed = all_checks_passed and passed

                        if "exp_avg_sq" in state:
                            passed = check_tensor_dtype(
                                state["exp_avg_sq"],
                                torch.float32,
                                "  Optimizer state 'exp_avg_sq'",
                            )
                            all_checks_passed = all_checks_passed and passed

                        state_checked = True
                        break  # Check just one parameter's optimizer state
                if state_checked:
                    break

            print(f"\n{'=' * 60}")
            if all_checks_passed:
                print("✅ ALL DTYPE CHECKS PASSED!")
                print("Parameters, gradients, and optimizer states are correctly in FP32")
            else:
                print("❌ SOME DTYPE CHECKS FAILED!")
                print("This indicates the mixed precision bug is present.")
                print("The fix is to set reduce_dtype=torch.float32 in MixedPrecisionPolicy")
            print(f"{'=' * 60}\n")

            # Assert all checks passed
            assert all_checks_passed, (
                "Mixed precision dtype checks failed! "
                "Model parameters, gradients, or optimizer states are not in FP32. "
                "This bug is fixed by setting reduce_dtype=torch.float32 in the "
                "MixedPrecisionPolicy in setup_model_for_training.py"
            )

            print(f"Test completed successfully - Loss: {loss.item():.4f}")

        finally:
            # Cleanup distributed environment
            if dist.is_initialized():
                dist.destroy_process_group()
