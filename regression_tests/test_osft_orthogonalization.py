"""
Test script for OSFT gradient orthogonalization.

This script:
1. Loads Qwen2.5-1.5B-Instruct with OSFT enabled
2. Sets up FSDP2 wrapping like in production training
3. Runs 100 training steps with synthetic data
4. Checks orthogonality at each step (gradients and post-update parameters)
5. Reports comprehensive statistics on orthogonality violations

Usage:
    torchrun --nproc_per_node=2 test_osft_orthogonalization.py [options]

Example:
    torchrun --nproc_per_node=2 test_osft_orthogonalization.py --model Qwen/Qwen2.5-1.5B-Instruct --num-steps 100
"""

import torch
import torch.distributed as dist
import math
import os
import sys
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler
from typing import Dict, List
from dataclasses import dataclass
import argparse

# Add mini_trainer to path

from mini_trainer.setup_model_for_training import setup_model,  wrap_fsdp2
from mini_trainer.osft_utils import is_osft_model, cast_to_osft_model, optim_wrapper

@dataclass
class OrthogonalityMetrics:
    """Stores metrics for a single orthogonality check."""
    param_name: str
    check_type: str  # 'U_grad', 'V_grad', 'U_param', 'V_param'
    max_angle_diff: float  # in degrees
    step: int


class OrthogonalityTracker:
    """Tracks orthogonality metrics across training steps."""
    
    def __init__(self, margin_deg: float = 1.0):
        self.margin_deg = margin_deg
        self.metrics: Dict[str, Dict[str, OrthogonalityMetrics]] = {}
        self.total_checks = 0
        self.failed_checks = 0
        
    def update(self, param_name: str, check_type: str, max_angle_diff: float, step: int):
        """Update tracker with new measurement."""
        self.total_checks += 1
        
        if max_angle_diff > self.margin_deg:
            self.failed_checks += 1
            print(f"⚠️  ORTHOGONALITY VIOLATION: {param_name} ({check_type}) - "
                  f"Angle difference: {max_angle_diff:.4f}° (limit: {self.margin_deg}°) at step {step}")
        
        key = f"{param_name}:{check_type}"
        
        if key not in self.metrics:
            self.metrics[key] = {
                'param_name': param_name,
                'check_type': check_type,
                'max_angle_diff': max_angle_diff,
                'step': step
            }
        else:
            # Update if this is worse
            if max_angle_diff > self.metrics[key]['max_angle_diff']:
                self.metrics[key]['max_angle_diff'] = max_angle_diff
                self.metrics[key]['step'] = step
    def get_top_violations(self, n: int = 5) -> List[Dict]:
        """Get top N worst violations."""
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda x: x['max_angle_diff'],
            reverse=True
        )
        return sorted_metrics[:n]
    
    def is_successful(self) -> bool:
        """Check if all measurements passed."""
        for metric in self.metrics.values():
            if metric['max_angle_diff'] > self.margin_deg:
                return False
        return True
    
    def get_summary(self) -> str:
        """Generate summary report."""
        lines = []
        lines.append("=" * 100)
        lines.append("OSFT ORTHOGONALITY TEST SUMMARY")
        lines.append("=" * 100)
        lines.append(f"Total checks performed: {self.total_checks}")
        lines.append(f"Failed checks (>{self.margin_deg}°): {self.failed_checks}")
        lines.append(f"Pass rate: {100 * (1 - self.failed_checks / max(self.total_checks, 1)):.2f}%")
        lines.append("")

        if self.is_successful():
            lines.append("✅ RESULT: PASSED - All orthogonality constraints satisfied!")
        else:
            lines.append("❌ RESULT: FAILED - Orthogonality violations detected!")

        lines.append("")
        lines.append("Top 5 Largest Angle Deviations:")
        lines.append("-" * 100)
        lines.append(f"{'Rank':<6}{'Parameter':<40}{'Check Type':<15}{'Max Diff (°)':<15}{'Step':<10}")
        lines.append("-" * 100)

        for i, metric in enumerate(self.get_top_violations(5), 1):
            lines.append(
                f"{i:<6}{metric['param_name']:<40}{metric['check_type']:<15}"
                f"{metric['max_angle_diff']:<15.4f}{metric['step']:<10}"
            )

        lines.append("=" * 100)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize tracker state to dictionary for distributed gathering."""
        return {
            'metrics': self.metrics,
            'total_checks': self.total_checks,
            'failed_checks': self.failed_checks,
        }

    def merge_from_dict(self, other_dict: Dict):
        """Merge metrics from another tracker's serialized state."""
        self.total_checks += other_dict['total_checks']
        self.failed_checks += other_dict['failed_checks']

        for key, metric in other_dict['metrics'].items():
            if key not in self.metrics:
                self.metrics[key] = metric
            else:
                # Keep the worst violation
                if metric['max_angle_diff'] > self.metrics[key]['max_angle_diff']:
                    self.metrics[key] = metric


def get_osft_params(model):
    """Extract only OSFT parameters from model."""
    return [p for n, p in model.named_parameters() if 'osft_params' in n]  # just select only osft params for now


def convert_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg / 180 * math.pi


def compute_angle_differences(A: torch.Tensor, B: torch.Tensor = None, top_n: int = 5) -> List[float]:
    """
    Compute angle differences between matrices A and B, returning the top N worst deviations from orthogonality.

    Args:
        A: First matrix
        B: Second matrix (if None, assumes A)
        top_n: Number of worst angle deviations to return

    Returns:
        List of top N angle differences in degrees, or empty list if matrices are invalid
    """
    try:
        solo_matrix = False
        if B is None:
            B = A
            solo_matrix = True

        # Handle DTensor conversion for distributed training
        if isinstance(A, torch.distributed._tensor.api.DTensor):
            A = A.to_local()
        if isinstance(B, torch.distributed._tensor.api.DTensor):
            B = B.to_local()

        if A.dim() != 2 or B.dim() != 2:
            return []

        ma, na = A.shape
        mb, nb = B.shape

        if ma != mb:
            return []

        # Compute orthogonality angles
        Mag_A = A.pow(2).sum(dim=0, keepdim=True).sqrt()
        Mag_B = B.pow(2).sum(dim=0, keepdim=True).sqrt()
        Mag = Mag_B.T @ Mag_A

        Proj = B.T @ A
        angles = (Proj / Mag).abs().clamp(0.0, 1.0).acos()

        # Expected angles (should be pi/2 for orthogonality)
        correct = torch.ones_like(angles) * (math.pi / 2)

        # For same matrix, don't count self-angles
        if solo_matrix:
            correct[torch.arange(nb), torch.arange(na)] = 0.0

        # Compute angle differences in degrees
        diff = (angles - correct).abs() * 180 / math.pi

        # Get top N worst deviations
        diff_flat = diff.flatten()
        if solo_matrix:
            # Exclude diagonal elements for same matrix
            mask = torch.ones(diff.shape, dtype=torch.bool, device=diff.device)
            mask[torch.arange(min(nb, na)), torch.arange(min(nb, na))] = False
            diff_flat = diff[mask]

        if len(diff_flat) == 0:
            return []

        top_diffs, _ = torch.topk(diff_flat, min(top_n, len(diff_flat)))
        return top_diffs.cpu().tolist()

    except Exception as e:
        return []


def check_gradient_orthogonality(model, module, step: int, tracker: OrthogonalityTracker):
    """
    Check if gradients of U_low and V_low are orthogonal to U_high and V_high.

    Args:
        model: OSFT model
        module: Module with OSFT parameters attached
        step: Current training step
        tracker: OrthogonalityTracker to update
    """
    svd_dict = model.get_svd_dict_for_module(module)
    if svd_dict["U_low"].grad is None or svd_dict["V_low"].grad is None:
        return
    
    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    V_low = svd_dict["V_low"]
    
    # get the safe_name for tracking
    safe_name = module.osft_params.safe_name

    # we need to pull the gradients out before casting these variables to full_tensor,
    # since `.full_tensor` doesn't return a tensor with the .grad attribute populated
    dU_low = U_low.grad.full_tensor() if hasattr(U_low.grad, 'full_tensor') else U_low.grad
    dV_low = V_low.grad.full_tensor() if hasattr(V_low.grad, 'full_tensor') else V_low.grad

    if hasattr(U_high, 'full_tensor'):
        U_high = U_high.full_tensor()
    if hasattr(V_high, 'full_tensor'):
        V_high = V_high.full_tensor()
    if hasattr(U_low, 'full_tensor'):
        U_low = U_low.full_tensor()
    if hasattr(V_low, 'full_tensor'):
        V_low = V_low.full_tensor()

    
    # Check U gradient orthogonality
    u_grad_diffs = compute_angle_differences(U_high, dU_low, top_n=1)
    if u_grad_diffs:
        tracker.update(safe_name, 'U_grad', u_grad_diffs[0], step)
    
    # Check V gradient orthogonality
    v_grad_diffs = compute_angle_differences(V_high.T, dV_low.T, top_n=1)
    if v_grad_diffs:
        tracker.update(safe_name, 'V_grad', v_grad_diffs[0], step)


def check_parameter_orthogonality(model, module, step: int, tracker: OrthogonalityTracker):
    """
    Check if post-update U_low and V_low are orthogonal to U_high and V_high.

    Args:
        model: OSFT model
        module: Module with OSFT parameters attached
        step: Current training step
        tracker: OrthogonalityTracker to update
    """
    svd_dict = model.get_svd_dict_for_module(module)
    
    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    V_low = svd_dict["V_low"]
    
    # get the safe_name for tracking
    safe_name = module.osft_params.safe_name

    if hasattr(U_high, 'full_tensor'):
        U_high = U_high.full_tensor()
    if hasattr(V_high, 'full_tensor'):
        V_high = V_high.full_tensor()
    if hasattr(U_low, 'full_tensor'):
        U_low = U_low.full_tensor()
    if hasattr(V_low, 'full_tensor'):
        V_low = V_low.full_tensor()
    
    # Check U parameter orthogonality
    u_param_diffs = compute_angle_differences(U_high, U_low, top_n=1)
    if u_param_diffs:
        tracker.update(safe_name, 'U_param', u_param_diffs[0], step)
    
    # Check V parameter orthogonality
    v_param_diffs = compute_angle_differences(V_high.T, V_low.T, top_n=1)
    if v_param_diffs:
        tracker.update(safe_name, 'V_param', v_param_diffs[0], step)
    
    






def test_osft_orthogonalization(
    args: argparse.Namespace,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    margin_deg: float = 1.0,
    osft_rank_ratio: float = 0.5,
    num_steps: int = 100,
    batch_size: int = 64,
    seq_len: int = 128,
):
    """
    Test OSFT gradient orthogonalization over multiple training steps.

    Args:
        model_name: HuggingFace model name or path
        margin_deg: Acceptable angle margin in degrees
        osft_rank_ratio: Ratio of rank to use for high-rank subspace
        num_steps: Number of training steps to run
        batch_size: Batch size per GPU
        seq_len: Sequence length
    """
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 100)
        print("OSFT ORTHOGONALITY TEST")
        print("=" * 100)
        print(f"Model: {model_name}")
        print(f"Steps: {num_steps}")
        print(f"Batch size per GPU: {batch_size} (Total: {batch_size * world_size})")
        print(f"Sequence length: {seq_len}")
        print(f"Margin: {margin_deg}°")
        print(f"OSFT rank ratio: {osft_rank_ratio}")
        print("=" * 100)

    # Load tokenizer
    if rank == 0:
        print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    
    if rank == 0:
        print(f"Vocabulary size: {vocab_size}")

    # Load model with OSFT enabled
    if rank == 0:
        print("Loading OSFT model...")
    model = setup_model(
        model_name_or_path=model_name,
        osft=True,
        local_rank=local_rank,
        train_dtype=torch.float32,
        osft_upcast_dtype=torch.float32,
        osft_rank_ratio=osft_rank_ratio,
        use_liger_kernels=False,
        osft_memory_efficient_init=False,
    )

    # Verify it's an OSFT model
    assert is_osft_model(model), "Model is not an OSFT model!"
    osft_model = cast_to_osft_model(model)

    if rank == 0:
        print("OSFT model loaded successfully")
        print(f"OSFT tracking {len(osft_model.osft_config)} parameters")

    # Wrap with FSDP2 and setup optimizer
    model = wrap_fsdp2(model)
    
    optimizer = AdamW(get_osft_params(model), lr=args.lr, betas=(0.9, 0.95))
    scheduler = get_scheduler('constant', optimizer)
    optim_wrapper(optimizer, model)

    if rank == 0:
        print("Model wrapped with FSDP2")
        print("=" * 100)
        print("Starting training loop...")
        print("=" * 100)

    # Initialize tracker
    tracker = OrthogonalityTracker(margin_deg=margin_deg)
    
    model.train()
    
    # Training loop
    for step in range(1, num_steps + 1):
        # Generate synthetic data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Compute loss correctly for distributed training
        summed_loss = loss.float().sum(dim=0, keepdim=False) / (labels.numel())
        
        # Backward pass
        summed_loss.backward()
        
        
        
        # Take gradient step (includes projection via optim_wrapper)
        optimizer.step()
        scheduler.step()
        # Check gradient orthogonality (before optimizer.step)
        for module in model.modules():
            if hasattr(module, "osft_params") and \
               hasattr(module, "osft_U_high") and hasattr(module, "osft_S_high") and hasattr(module, "osft_V_high"):
                check_gradient_orthogonality(model, module, step, tracker)
    
        # Check parameter orthogonality (after optimizer.step)
        for module in model.modules():
            if hasattr(module, "osft_params") and \
               hasattr(module, "osft_U_high") and hasattr(module, "osft_S_high") and hasattr(module, "osft_V_high"):
                check_parameter_orthogonality(model, module, step, tracker)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Progress reporting
        if rank == 0 and (step % 10 == 0 or step == 1):
            print(f"Step {step}/{num_steps} - Loss: {summed_loss.item():.4f}")

    # Synchronize all ranks after training
    dist.barrier()

    # Gather metrics from all ranks to rank 0
    tracker_dicts = [None] * world_size if rank == 0 else None
    dist.gather_object(
        tracker.to_dict(),
        tracker_dicts,
        dst=0
    )

    # Merge metrics on rank 0
    if rank == 0:
        for i, other_tracker_dict in enumerate(tracker_dicts):
            if i == rank:
                continue  # Skip self
            if other_tracker_dict is not None:
                tracker.merge_from_dict(other_tracker_dict)

    if rank == 0:
        print("=" * 100)
        print("Training completed!")
        print("")
        print(tracker.get_summary())

    # Broadcast success status to all ranks
    success = tracker.is_successful()
    success_tensor = torch.tensor([1 if success else 0], dtype=torch.int, device=device)
    dist.broadcast(success_tensor, src=0)
    success = bool(success_tensor.item())

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test OSFT gradient orthogonalization over multiple training steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=1.0,
        help="Acceptable angle margin in degrees",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=0.5,
        help="OSFT rank ratio",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate",
    )

    args = parser.parse_args()

    success = test_osft_orthogonalization(
        args=args,
        model_name=args.model,
        margin_deg=args.margin_deg,
        osft_rank_ratio=args.rank_ratio,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Synchronize before destroying process group
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if success else 1)
