"""Tests for accelerator selection in mini_trainer.device_utils."""

from unittest.mock import patch

import pytest
import torch

from mini_trainer import device_utils


@pytest.fixture(autouse=True)
def _unpinned(monkeypatch):
    # conftest pins unit tests to the CUDA path; these tests exercise selection.
    monkeypatch.delenv("MINI_TRAINER_DEVICE", raising=False)


def _fake(monkeypatch, *, cuda: bool, xpu: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)


@pytest.mark.parametrize(
    "cuda,xpu,expected_type,expected_backend",
    [
        (True, False, "cuda", "nccl"),
        (True, True, "cuda", "nccl"),  # CUDA wins when both are present
        (False, True, "xpu", "xccl"),
        (False, False, "cuda", "nccl"),  # CPU-only hosts keep the original CUDA path
    ],
)
def test_selection(monkeypatch, cuda, xpu, expected_type, expected_backend):
    _fake(monkeypatch, cuda=cuda, xpu=xpu)
    assert device_utils.get_device_type() == expected_type
    assert device_utils.get_dist_backend() == expected_backend
    assert device_utils.get_device_module() is getattr(torch, expected_type)
    assert device_utils.get_device(3) == torch.device(expected_type, 3)


def test_env_override(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("MINI_TRAINER_DEVICE", "CUDA")
    assert device_utils.get_device_type() == "cuda"


def test_get_device_defaults_to_local_rank(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert device_utils.get_device() == torch.device("xpu", 2)


def test_helpers_dispatch_to_xpu_module(monkeypatch):
    _fake(monkeypatch, cuda=False, xpu=True)
    calls = []
    for name in ("empty_cache", "reset_peak_memory_stats", "set_device", "manual_seed_all"):
        monkeypatch.setattr(torch.xpu, name, lambda *a, _n=name: calls.append((_n, a)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CUDA used on XPU host"))
    device_utils.empty_cache()
    device_utils.reset_peak_memory_stats()
    device_utils.set_device(1)
    device_utils.manual_seed_all(7)
    assert calls == [
        ("empty_cache", ()),
        ("reset_peak_memory_stats", ()),
        ("set_device", (1,)),
        ("manual_seed_all", (7,)),
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="Intel XPU not available")
def test_real_xpu_selected_when_no_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; CUDA is preferred")
    assert device_utils.get_device_type() == "xpu"
    t = torch.ones(4, device=device_utils.get_device(0))
    assert t.device.type == "xpu"
    assert device_utils.memory_allocated() > 0


@pytest.fixture
def _fresh_attn_choice():
    device_utils.xpu_flash_attn_implementation.cache_clear()
    yield
    device_utils.xpu_flash_attn_implementation.cache_clear()


@pytest.mark.usefixtures("_fresh_attn_choice")
def test_xpu_attn_uses_hub_flash_kernel():
    with patch("transformers.integrations.hub_kernels.load_and_register_attn_kernel") as load:
        assert device_utils.xpu_flash_attn_implementation() == device_utils.XPU_FLASH_ATTN_KERNEL
    load.assert_called_once_with(device_utils.XPU_FLASH_ATTN_KERNEL)


@pytest.mark.usefixtures("_fresh_attn_choice")
def test_xpu_attn_falls_back_to_sdpa_when_kernel_unavailable(caplog):
    with patch(
        "transformers.integrations.hub_kernels.load_and_register_attn_kernel",
        side_effect=ImportError("kernels is not installed"),
    ):
        assert device_utils.xpu_flash_attn_implementation() == "sdpa"
    assert "using SDPA instead" in caplog.text


@pytest.mark.gpu
@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="Intel XPU not available")
@pytest.mark.usefixtures("_fresh_attn_choice")
def test_packed_sequences_isolated_with_xpu_attention():
    # Padding-free batches pack samples into one row; attention must not cross
    # sample boundaries. Compare against running each sample on its own.
    from transformers import LlamaConfig, LlamaForCausalLM

    impl = device_utils.xpu_flash_attn_implementation()
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attn_implementation=impl,
    )
    model = LlamaForCausalLM(cfg).to("xpu", torch.bfloat16).eval()
    assert model.config._attn_implementation == impl
    lens = [7, 19, 4]
    seqs = [torch.randint(0, 128, (n,), device="xpu") for n in lens]
    ids = torch.cat(seqs)[None]
    pos = torch.cat([torch.arange(n, device="xpu") for n in lens])[None]
    with torch.no_grad():
        separate = torch.cat([model(s[None]).logits[0] for s in seqs]).float()
        packed = model(ids, position_ids=pos, use_cache=False).logits[0].float()
        leaky = model(ids, use_cache=False).logits[0].float()  # no isolation
    assert (packed - separate).abs().max() < 0.05, impl
    assert (leaky - separate).abs().max() > 0.1  # the check can see leakage
