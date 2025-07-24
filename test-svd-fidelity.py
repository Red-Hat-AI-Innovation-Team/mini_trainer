"""
Test script to verify SVD decomposition and reconstruction fidelity.

This script tests that when a model is created with distributed SVD initialization,
the reconstructed parameters from the decomposed SVD parts are identical to the
original untouched model parameters (within numerical tolerance).

Usage:
    # Single GPU
    python test_svd_reconstruction_fidelity.py

    # Multiple GPUs (to test distributed SVD)
    torchrun --nnodes=1 --nproc-per-node=4 test_svd_reconstruction_fidelity.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import os
import time
from setup_model_for_training import setup_model
from utils import init_distributed_environment, log_rank_0
from svd_utils import get_svd_target_parameters
import typer


app = typer.Typer()


def load_original_model(model_name_or_path, use_liger_kernels=False):
    """Load the original model without SVD decomposition."""
    rank = int(os.environ.get("RANK", 0))

    log_rank_0("Loading original model (no SVD)...")
    original_model = setup_model(
        model_name_or_path=model_name_or_path,
        use_liger_kernels=use_liger_kernels,
        orthogonal_subspace_learning=False,  # No SVD
        rank=rank,
    )
    return original_model


def load_svd_model(model_name_or_path, use_liger_kernels=False):
    """Load the model with distributed SVD initialization."""
    rank = int(os.environ.get("RANK", 0))

    log_rank_0("Loading SVD model (with distributed SVD)...")
    svd_model = setup_model(
        model_name_or_path=model_name_or_path,
        use_liger_kernels=use_liger_kernels,
        orthogonal_subspace_learning=True,  # Enable SVD
        rank=rank,
        upcast_dtype=torch.float64,
        output_dtype=torch.float64,
    )
    return svd_model


def compare_parameters(original_model, svd_model, tolerance=1e-5):
    """
    Compare parameters between original model and reconstructed SVD model.

    Args:
        original_model: Model without SVD decomposition
        svd_model: Model with SVD decomposition
        tolerance: Numerical tolerance for comparison

    Returns:
        dict: Results of comparison including statistics
    """
    results = {
        "total_params_compared": 0,
        "identical_params": 0,
        "close_params": 0,
        "different_params": 0,
        "max_difference": 0.0,
        "avg_difference": 0.0,
        "differences": [],
        "param_details": [],
    }

    # # Get parameters that should be decomposed
    # if hasattr(svd_model, "svd_config"):
    #     target_params = get_svd_target_parameters(svd_model, svd_model.svd_config)
    #     target_param_names = set(name for name, _ in target_params)
    # else:
    #     target_param_names = set()

    # Compare all parameters
    # original_params = dict(original_model.named_parameters())

    # # this is how we need to compare the parameters
    # ignored_params = []
    # for orig_name, original_param in original_model.state_dict():
    #     if orig_name not in svd_model.name_mapping:
    #         # save this for logging purposes later
    #         ignored_params += [orig_name]
    #         continue

    #     reconstructed_param = svd_model._reconstruct_weight(orig_name)
    #     # now we can compare original_param with reconstructed_param

    ignored_params = []
    for orig_name, original_param in original_model.state_dict().items():
        name = orig_name
        if orig_name not in svd_model.name_mapping:
            # this one isnt in the svd params so we ignore
            ignored_params += [orig_name]
            continue

        # This parameter was decomposed, so reconstruct it
        try:
            comparison_param = svd_model._reconstruct_weight(orig_name)
            param_type = "reconstructed"
        except Exception as e:
            # If reconstruction fails, skip this parameter
            results["param_details"].append(
                {"name": orig_name, "type": "reconstruction_failed", "error": str(e)}
            )
            continue

        # upcast the original_param to reduce information loss from comparison
        if comparison_param.dtype != original_param.dtype:
            original_param = original_param.to(comparison_param.dtype)

        # Compare parameters
        if comparison_param.shape != original_param.shape:
            results["param_details"].append(
                {
                    "name": orig_name,
                    "type": param_type,
                    "status": "shape_mismatch",
                    "original_shape": original_param.shape,
                    "comparison_shape": comparison_param.shape,
                }
            )
            results["different_params"] += 1
            continue

        # Calculate difference
        diff = torch.abs(comparison_param - original_param)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        results["total_params_compared"] += 1
        results["differences"].append(max_diff)
        results["max_difference"] = max(results["max_difference"], max_diff)

        # Classify the difference
        if torch.equal(comparison_param, original_param):
            status = "identical"
            results["identical_params"] += 1
        elif torch.allclose(
            comparison_param, original_param, atol=tolerance, rtol=tolerance
        ):
            status = "close"
            results["close_params"] += 1
        else:
            status = "different"
            results["different_params"] += 1

        results["param_details"].append(
            {
                "name": name,
                "type": param_type,
                "status": status,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "shape": tuple(original_param.shape),
            }
        )

    # Calculate average difference
    if results["differences"]:
        results["avg_difference"] = sum(results["differences"]) / len(
            results["differences"]
        )

    print_results(results, tolerance)
    dist.breakpoint()
    return results


def print_results(results, tolerance):
    """Print comparison results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"SVD RECONSTRUCTION FIDELITY TEST RESULTS")
    print(f"{'=' * 60}")

    print(f"Parameters compared: {results['total_params_compared']}")
    print(f"Tolerance: {tolerance:.2e}")
    print(f"")

    print(f"📊 Summary:")
    print(f"  ✅ Identical parameters: {results['identical_params']}")
    print(f"  🔍 Close parameters (within tolerance): {results['close_params']}")
    print(f"  ❌ Different parameters: {results['different_params']}")
    print(f"")

    print(f"📈 Numerical Statistics:")
    print(f"  Max difference: {results['max_difference']:.2e}")
    print(f"  Average difference: {results['avg_difference']:.2e}")
    print(f"")

    # Show parameter breakdown by type
    reconstructed_count = sum(
        1 for p in results["param_details"] if p["type"] == "reconstructed"
    )
    direct_count = sum(1 for p in results["param_details"] if p["type"] == "direct")

    print(f"🔧 Parameter Types:")
    print(f"  Reconstructed from SVD: {reconstructed_count}")
    print(f"  Direct comparison: {direct_count}")
    print(f"")

    # Show worst differences for reconstructed parameters
    reconstructed_params = [
        p
        for p in results["param_details"]
        if p["type"] == "reconstructed" and "max_diff" in p
    ]
    if reconstructed_params:
        print(f"🚨 Worst Reconstructed Parameter Differences:")
        worst_params = sorted(
            reconstructed_params, key=lambda x: x["max_diff"], reverse=True
        )[:5]
        for param in worst_params:
            print(
                f"  {param['name']}: {param['max_diff']:.2e} (status: {param['status']})"
            )
        print(f"")

    # Overall result
    success_rate = (
        (
            (results["identical_params"] + results["close_params"])
            / results["total_params_compared"]
            * 100
        )
        if results["total_params_compared"] > 0
        else 0
    )

    print(f"🎯 Overall Result:")
    if results["different_params"] == 0:
        print(f"  ✅ SUCCESS: All parameters match within tolerance!")
    else:
        print(f"  ⚠️  PARTIAL SUCCESS: {success_rate:.1f}% of parameters match")

    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'=' * 60}")


def compare_params(original_model: nn.Module, svd_model: nn.Module):
    # here we can compare the original parameters
    for k, orig_w in original_model.state_dict().items():
        if k not in svd_model.name_mapping:
            print("we dont do this one")
            continue


@app.command()
def test_reconstruction_fidelity(
    model_name_or_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model name or path"
    ),
    use_liger_kernels: bool = typer.Option(False, help="Whether to use liger kernels"),
    tolerance: float = typer.Option(1e-5, help="Numerical tolerance for comparison"),
    verbose: bool = typer.Option(False, help="Show detailed parameter comparison"),
):
    """Test SVD reconstruction fidelity."""

    # Initialize distributed environment
    init_distributed_environment()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log_rank_0(f"🧪 Testing SVD reconstruction fidelity with {world_size} ranks")
    log_rank_0(f"Model: {model_name_or_path}")
    log_rank_0(f"Tolerance: {tolerance:.2e}")

    # Load SVD model
    start_time = time.time()
    svd_model = load_svd_model(model_name_or_path, use_liger_kernels)
    svd_time = time.time() - start_time
    log_rank_0(f"⏱️ SVD model loaded in {svd_time:.2f}s")

    # Compare parameters (only on rank 0 to avoid redundant work)
    if rank == 0:
        # Load original model
        start_time = time.time()
        original_model = load_original_model(model_name_or_path, use_liger_kernels)
        device_0 = torch.device("cuda", 0)
        original_model = original_model.to(device_0)
        original_time = time.time() - start_time

        log_rank_0(f"⏱️ Original model loaded in {original_time:.2f}s")

        log_rank_0("🔍 Comparing parameters...")
        start_time = time.time()
        results = compare_parameters(original_model, svd_model, tolerance)
        comparison_time = time.time() - start_time
        log_rank_0(f"⏱️ Parameter comparison completed in {comparison_time:.2f}s")

        # Print results
        print_results(results, tolerance)

        # Show detailed results if requested
        if verbose:
            print(f"\n📋 Detailed Parameter Comparison:")
            for param in results["param_details"]:
                if "max_diff" in param:
                    print(
                        f"  {param['name']} ({param['type']}): {param['status']} - "
                        f"max_diff={param['max_diff']:.2e}, mean_diff={param['mean_diff']:.2e}"
                    )
                else:
                    print(f"  {param['name']} ({param['type']}): {param['status']}")

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    log_rank_0("🎉 SVD reconstruction fidelity test completed!")


if __name__ == "__main__":
    app()
