"""GPU tests for per-block torch.compile integration with FSDP2."""

import os

os.environ["TESTING"] = "true"

import gc

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from mini_trainer.setup_model_for_training import setup_model, setup_training_components
from mini_trainer.utils import patch_target_module


def create_tiny_llama_model():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        hidden_act="silu",
    )
    return LlamaForCausalLM(config), config


def _run_steps(model_path, compile_model, input_ids, labels, num_steps=3, osft=False, osft_rank_ratio=0.25):
    """Load model, wrap with FSDP2 (± compile), run steps, return losses."""
    model = setup_model(
        model_name_or_path=str(model_path),
        use_liger_kernels=False,
        osft=osft,
        osft_rank_ratio=osft_rank_ratio if osft else None,
        local_rank=0,
    )
    model, optimizer, lr_scheduler = setup_training_components(
        model,
        learning_rate=1e-3,
        num_warmup_steps=0,
        lr_scheduler="constant",
        compile_model=compile_model,
    )

    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss.float().sum() / input_ids.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

    return losses, model


@pytest.mark.gpu
class TestCompile:
    @pytest.fixture(autouse=True, scope="class")
    def dist_env(self):
        """Single process group for the entire test class."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

        from mini_trainer.none_reduction_losses import (
            hf_fixed_cross_entropy_none_reduction,
        )

        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )

        yield

        dist.destroy_process_group()

    @pytest.fixture(autouse=True)
    def reset_dynamo(self):
        torch._dynamo.reset()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False
        yield
        torch._dynamo.reset()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False

    @pytest.fixture
    def saved_model(self, tmp_path):
        """Create and save a tiny Llama model + tokenizer to disk."""
        torch.manual_seed(42)
        model, config = create_tiny_llama_model()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model_path = tmp_path / "tiny_llama"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        return model_path, config

    def test_compiled_matches_eager(self, saved_model, single_gpu_device):
        """Compiled and eager produce approximately equal losses.

        Not exact equality: bf16 mixed precision + inductor kernel fusion
        reorder floating-point ops, so small divergence is expected.
        Observed gap is O(1e-3) on loss values O(1e+2), growing across
        steps as differences accumulate. Tolerance is set at 10x the
        observed per-step gap.
        """
        model_path, config = saved_model

        torch.manual_seed(99)
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=single_gpu_device)
        labels = input_ids.clone()

        # Eager run
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        eager_losses, eager_model = _run_steps(model_path, compile_model=False, input_ids=input_ids, labels=labels)

        del eager_model
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

        # Compiled run (same model weights from disk, same seed)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        compiled_losses, _ = _run_steps(model_path, compile_model=True, input_ids=input_ids, labels=labels)

        for step, (e, c) in enumerate(zip(eager_losses, compiled_losses)):
            assert abs(e - c) < 0.1, (
                f"Step {step}: eager loss {e:.6f} vs compiled loss {c:.6f}, diff {abs(e - c):.2e} exceeds tolerance 0.1"
            )

    def test_no_graph_breaks_and_dynamic_shapes(self, saved_model, single_gpu_device):
        """Forward/backward completes under fullgraph=True with varied seq_len.

        Two forward/backward passes with different sequence lengths verify:
        1. No graph breaks (fullgraph=True contract)
        2. dynamic=True reuses the same compiled graph (no recompilation)
        """
        model_path, config = saved_model

        torch._dynamo.utils.counters.clear()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

        model = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=False,
            local_rank=0,
        )
        model, optimizer, lr_scheduler = setup_training_components(
            model,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=True,
        )

        # Step 1: seq_len=32
        input_ids_1 = torch.randint(0, config.vocab_size, (2, 32), device=single_gpu_device)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids_1, labels=input_ids_1.clone()).loss.float().sum()
        loss.backward()
        optimizer.step()

        compilations_after_first = torch._dynamo.utils.counters["stats"]["ok"]

        # Step 2: seq_len=48 (different shape — should reuse graph via dynamic=True)
        input_ids_2 = torch.randint(0, config.vocab_size, (2, 48), device=single_gpu_device)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids_2, labels=input_ids_2.clone()).loss.float().sum()
        loss.backward()
        optimizer.step()

        compilations_after_second = torch._dynamo.utils.counters["stats"]["ok"]

        graph_breaks = dict(torch._dynamo.utils.counters["graph_break"])
        assert len(graph_breaks) == 0, f"Graph breaks detected: {graph_breaks}"

        assert compilations_after_second == compilations_after_first, (
            f"dynamic=True should prevent recompilation on shape change, "
            f"but compilations went from {compilations_after_first} to {compilations_after_second}"
        )

    def test_optimized_module_wrappers(self, saved_model, single_gpu_device):
        """Compiled blocks are OptimizedModule; uncompiled blocks are not."""
        model_path, _ = saved_model

        # Compiled path
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        model_c = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=False,
            local_rank=0,
        )
        model_c, _, _ = setup_training_components(
            model_c,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=True,
        )

        from torch._dynamo.eval_frame import OptimizedModule

        layers_c = model_c.model.layers
        for idx, block in enumerate(layers_c):
            assert isinstance(block, OptimizedModule), f"Block {idx} should be OptimizedModule, got {type(block)}"

        del model_c
        gc.collect()
        torch.cuda.empty_cache()

        # Eager path
        model_e = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=False,
            local_rank=0,
        )
        model_e, _, _ = setup_training_components(
            model_e,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=False,
        )

        layers_e = model_e.model.layers
        for idx, block in enumerate(layers_e):
            assert not isinstance(block, OptimizedModule), (
                f"Block {idx} should NOT be OptimizedModule, got {type(block)}"
            )

    def test_compile_works_without_dynamo_config_flag(self, saved_model, single_gpu_device):
        """AC + compile works without skip_fwd_side_effects_in_bwd_under_checkpoint.

        The flag is set defensively in train.py but is not required on current
        PyTorch. If this test starts failing on a future version, the flag
        becomes load-bearing and the comment in train.py should be updated.
        """
        model_path, config = saved_model

        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False

        model = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=False,
            local_rank=0,
        )
        model, optimizer, _ = setup_training_components(
            model,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=True,
        )

        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=single_gpu_device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss.float().sum()
        loss.backward()
        optimizer.step()


@pytest.mark.gpu
class TestOSFTCompile:
    @pytest.fixture(autouse=True, scope="class")
    def dist_env(self):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12357"
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

        from mini_trainer.none_reduction_losses import (
            hf_fixed_cross_entropy_none_reduction,
        )

        patch_target_module(
            "transformers.loss.loss_utils.fixed_cross_entropy",
            hf_fixed_cross_entropy_none_reduction,
        )

        yield

        dist.destroy_process_group()

    @pytest.fixture(autouse=True)
    def reset_dynamo(self):
        torch._dynamo.reset()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False
        yield
        torch._dynamo.reset()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = False

    @pytest.fixture
    def saved_model(self, tmp_path):
        torch.manual_seed(42)
        model, config = create_tiny_llama_model()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model_path = tmp_path / "tiny_llama"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        return model_path, config

    def test_osft_compiled_matches_eager(self, saved_model, single_gpu_device):
        """Compiled and eager OSFT produce approximately equal losses."""
        model_path, config = saved_model

        torch.manual_seed(99)
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device=single_gpu_device)
        labels = input_ids.clone()

        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        eager_losses, eager_model = _run_steps(
            model_path,
            compile_model=False,
            input_ids=input_ids,
            labels=labels,
            osft=True,
        )

        del eager_model
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        compiled_losses, _ = _run_steps(
            model_path,
            compile_model=True,
            input_ids=input_ids,
            labels=labels,
            osft=True,
        )

        for step, (e, c) in enumerate(zip(eager_losses, compiled_losses)):
            assert abs(e - c) < 0.1, (
                f"Step {step}: eager loss {e:.6f} vs compiled loss {c:.6f}, diff {abs(e - c):.2e} exceeds tolerance 0.1"
            )

    def test_osft_no_graph_breaks(self, saved_model, single_gpu_device):
        """OSFT forward/backward completes under fullgraph=True with varied seq_len."""
        model_path, config = saved_model

        torch._dynamo.utils.counters.clear()
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

        model = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=True,
            osft_rank_ratio=0.25,
            local_rank=0,
        )
        model, optimizer, lr_scheduler = setup_training_components(
            model,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=True,
        )

        input_ids_1 = torch.randint(0, config.vocab_size, (2, 32), device=single_gpu_device)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids_1, labels=input_ids_1.clone()).loss.float().sum()
        loss.backward()
        optimizer.step()

        compilations_after_first = torch._dynamo.utils.counters["stats"]["ok"]

        input_ids_2 = torch.randint(0, config.vocab_size, (2, 48), device=single_gpu_device)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids_2, labels=input_ids_2.clone()).loss.float().sum()
        loss.backward()
        optimizer.step()

        compilations_after_second = torch._dynamo.utils.counters["stats"]["ok"]

        graph_breaks = dict(torch._dynamo.utils.counters["graph_break"])
        assert len(graph_breaks) == 0, f"Graph breaks detected: {graph_breaks}"

        assert compilations_after_second == compilations_after_first, (
            f"dynamic=True should prevent recompilation on shape change, "
            f"but compilations went from {compilations_after_first} to {compilations_after_second}"
        )

    def test_osft_optimized_module_wrappers(self, saved_model, single_gpu_device):
        """Compiled OSFT blocks are OptimizedModule."""
        model_path, _ = saved_model

        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        model = setup_model(
            model_name_or_path=str(model_path),
            use_liger_kernels=False,
            osft=True,
            osft_rank_ratio=0.25,
            local_rank=0,
        )
        model, _, _ = setup_training_components(
            model,
            learning_rate=1e-3,
            num_warmup_steps=0,
            lr_scheduler="constant",
            compile_model=True,
        )

        from torch._dynamo.eval_frame import OptimizedModule

        layers = model.model.layers
        for idx, block in enumerate(layers):
            assert isinstance(block, OptimizedModule), f"Block {idx} should be OptimizedModule, got {type(block)}"
