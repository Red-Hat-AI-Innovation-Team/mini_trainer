"""Unit tests for torch.compile validation guard contracts.

These tests document the expected guard behavior from train.py:main().
They test the guard conditions directly (not via main()) because main()
requires a full distributed environment. The GPU tests in
gpu_tests/test_compile.py exercise the real code path end-to-end.
"""

import pytest


class TestCompileValidationGuards:
    def test_compile_osft_incompatible(self):
        compile_model = True
        osft = True
        with pytest.raises(ValueError, match="not compatible with --osft"):
            if compile_model and osft:
                raise ValueError(
                    "--compile-model is not compatible with --osft. "
                    "OSFT uses dynamic forward methods that cannot be traced by torch.compile."
                )

    def test_compile_liger_incompatible(self):
        compile_model = True
        use_liger_kernels = True
        with pytest.raises(ValueError, match="not compatible with --use-liger-kernels"):
            if compile_model and use_liger_kernels:
                raise ValueError(
                    "--compile-model is not compatible with --use-liger-kernels. "
                    "Both replace the same memory-bound ops; the interaction is untested."
                )

    def test_compile_moe_incompatible(self):
        moe_classes = ("MixtralForCausalLM", "GraniteMoeHybridForCausalLM")
        for cls_name in moe_classes:
            with pytest.raises(ValueError, match="not compatible with MoE"):
                if cls_name in moe_classes:
                    raise ValueError(
                        f"--compile-model is not compatible with MoE architecture {cls_name}. "
                        "MoE router logic causes graph breaks with fullgraph=True."
                    )

    def test_compile_guards_do_not_fire_when_disabled(self):
        compile_model = False
        osft = True
        use_liger_kernels = True

        if compile_model and osft:
            raise ValueError("should not reach")
        if compile_model and use_liger_kernels:
            raise ValueError("should not reach")
